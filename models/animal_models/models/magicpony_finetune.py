from types import SimpleNamespace
import numpy as np
import os
import cv2
import torch
import sys
from einops import rearrange, repeat
from torch.nn import functional as F
from itertools import chain


submodule_dir = os.path.abspath("externals/Animals")
sys.path.append(submodule_dir)
from models.animal_models.utils import *
from externals.Animals.model.models import AnimalModel, AnimalModelConfig
from externals.Animals.model.predictors import BasePredictorBase, InstancePredictorBase


class MagicponyFinetune(AnimalModel):
    def __init__(self, cfg: AnimalModelConfig):
        super().__init__(cfg)
        self.netBase = BasePredictorBase(self.cfg_predictor_base)
        self.netInstance = InstancePredictorBase(self.cfg_predictor_instance)

    def set_train(self):
        super().set_train()
        # Freeze everything
        for param in chain(self.netInstance.parameters(), self.netBase.parameters()):
            param.requires_grad = False
        if self.netInstance.cfg_articulation.enable_refine:
            # Unfreeze netArticulationRefine
            for param in self.netInstance.netArticulationRefine.parameters():
                param.requires_grad = True
            # Unfreeze netDepthEncoder
            for param in self.netInstance.netDepthEncoder.parameters():
                param.requires_grad = True
        else:
            # Unfreeze netArticulation
            for param in chain(self.netInstance.netArticulation.parameters()):
                param.requires_grad = True
            # if self.netInstance.cfg_articulation.use_depth_feature:
            #     # Unfreeze depth_feat_gate_param and depth_feat_proj
            #     for param in chain(self.netInstance.depth_feat_proj.parameters()):
            #         param.requires_grad = True
            #     self.netInstance.depth_feat_gate_param.requires_grad = True

    def set_train_post_load(self):
        if self.netInstance.cfg_articulation.enable_refine:
            # copy netArticulation to netArticulationRefine
            self.netInstance.netArticulationRefine.load_state_dict(
                self.netInstance.netArticulation.state_dict(),
                strict=False
            )

    def compute_regularizers(
        self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None, prior_shape=None, mvp=None,
        mask_gt=None, keypoint_gt=None, keypoint_gt_flag=None, **kwargs
    ):
        losses, aux = super().compute_regularizers(
            arti_params=arti_params, deformation=deformation, pose_raw=pose_raw, posed_bones=posed_bones, mvp=mvp,
            prior_shape=prior_shape, mask_gt=mask_gt
        )
        # Keypoint projection loss
        if keypoint_gt is not None and posed_bones is not None and keypoint_gt_flag.any():
            assert mvp is not None and mask_gt is not None
            pred_to_gt_map = {8: 7, 9: 6, 10: 5, 11: 13, 12: 12, 13: 11, 14: 16, 15: 15, 16: 14, 17: 10, 18: 9, 19: 8}
            bone_world4 = torch.concat(
                [posed_bones, torch.ones_like(posed_bones[..., :1]).to(posed_bones.device)], dim=-1
            )
            b, f, num_bones = bone_world4.shape[:3]
            bones_clip4 = (
                bone_world4.view(b, f, num_bones * 2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)
            ).view(b, f, num_bones, 2, 4)
            bones_uvd = bones_clip4[..., :3] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 3
            keypoint_pred = bones_uvd[:, :, list(pred_to_gt_map.keys()), -1, :2]  # b, f, k, 2
            keypoint_gt = keypoint_gt[:, :, list(pred_to_gt_map.values()), :2]  # b, f, k, 2
            losses["keypoint_projection_loss"] = ((keypoint_pred - keypoint_gt)[keypoint_gt_flag] ** 2).mean()
        return losses, aux

    def forward(
        self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None,
        logger_prefix='', is_training=True
    ):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, keypoint_gt, seq_idx, frame_idx = batch
        global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.dataset.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.dataset.out_image_size
        aux_viz = {}

        dino_feat_im_gt = None if dino_feat_im is None else expandBF(
            torch.nn.functional.interpolate(collapseBF(dino_feat_im), size=[h, w], mode="bilinear"), batch_size,
            num_frames)[:, :, :self.cfg_predictor_base.cfg_dino.feature_dim]
        dino_cluster_im_gt = None if dino_cluster_im is None else expandBF(
            torch.nn.functional.interpolate(collapseBF(dino_cluster_im), size=[h, w], mode="nearest"), batch_size,
            num_frames)

        ## GT image
        image_gt = input_image
        # if self.dataset.out_image_size != self.dataset.in_image_size:
        #     image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'),
        #                         batch_size, num_frames)
        #     if flow_gt is not None:
        #         flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"),
        #                            batch_size, num_frames - 1)

        ## predict prior shape and DINO
        if in_range(total_iter, self.cfg_predictor_base.cfg_shape.grid_res_coarse_iter_range):
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res_coarse
        else:
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res
        if self.get_predictor("netBase").netShape.grid_res != grid_res:
            self.get_predictor("netBase").netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net = self.netBase(total_iter=total_iter, is_training=is_training)
        else:
            prior_shape, dino_net = self.netBase(total_iter=total_iter, is_training=is_training)

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                    input_image, prior_shape, epoch, total_iter, is_training=is_training, frame_ids=global_frame_id,
                )  # first two dim dimensions already collapsed N=(B*F
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                input_image, prior_shape, epoch, total_iter, is_training=is_training, frame_ids=global_frame_id,
            )  # first two dim dimensions already collapsed N=(B*F)
        keypoint_gt_flag = disable_keypoint_loss(w2c, b=batch_size)
        keypoint_gt_flag = rearrange(keypoint_gt_flag, "(b f) -> b f", b=batch_size, f=num_frames)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)
        final_losses = {}

        ## render images
        if self.enable_render or not is_training:  # Force render for val and test
            render_flow = self.cfg_render.render_flow and num_frames > 1
            # render_modes = ['shaded', 'dino_pred', 'depth']
            render_modes = ['shaded', 'dino_pred']
            if render_flow:
                render_modes += ['flow']
            render_mvp, render_w2c, render_campos = (mvp, w2c, campos) if not self.cfg_render.render_default else (
            self.default_mvp.to(self.device), self.default_w2c.to(self.device), self.default_campos.to(self.device))
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    renders = self.render(
                        render_modes, shape, texture, render_mvp, render_w2c, render_campos, (h, w),
                        im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                    )
            else:
                renders = self.render(
                    render_modes, shape, texture, render_mvp, render_w2c, render_campos, (h, w),
                    im_features=im_features, light=light,
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                )
            if batch_size * num_frames != renders[0].shape[0]:
                batch_size = int(renders[0].shape[0] / num_frames)
            renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
            if render_flow:  # TODO: modify to use dict get instead of list/tuple
                shaded, dino_feat_im_pred, flow_pred = renders
                flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW
            else:
                shaded, dino_feat_im_pred = renders
                flow_pred = None
            image_pred = shaded[:, :, :3]
            mask_pred = shaded[:, :, 3]

            ## compute reconstruction losses
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    losses = self.compute_reconstruction_losses(
                        image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt,
                        dino_feat_im_gt, dino_feat_im_pred, mvp=mvp, posed_bones=forward_aux.get("posed_bones"),
                        background_mode=self.cfg_render.background_mode, reduce=False
                    )
            else:
                losses = self.compute_reconstruction_losses(
                    image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                    dino_feat_im_pred, mvp=mvp, posed_bones=forward_aux.get("posed_bones"),
                    background_mode=self.cfg_render.background_mode, reduce=False
                )

            ## supervise the rotation logits directly with reconstruction loss
            logit_loss_target = None
            if losses is not None:
                logit_loss_target = torch.zeros_like(expandBF(rot_logit, batch_size, num_frames))
                for name, loss in losses.items():
                    loss_weight = getattr(self.cfg_loss, f"{name}_weight")
                    if name in ['dino_feat_im_loss']:
                        ## increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier
                    if loss_weight > 0:
                        logit_loss_target += loss * loss_weight

                    ## multiply the loss with probability of the rotation hypothesis (detached)
                    if self.get_predictor("netInstance").cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
                        loss_prob = rot_prob.detach().view(batch_size, num_frames)[:,
                                    :loss.shape[1]]  # handle edge case for flow loss with one frame less
                        loss = loss * loss_prob * self.get_predictor("netInstance").num_pose_hypos
                    ## only compute flow loss for frames with the same rotation hypothesis
                    if name == 'flow_loss' and num_frames > 1:
                        ri = rot_idx.view(batch_size, num_frames)
                        same_rot_idx = (ri[:, 1:] == ri[:, :-1]).float()
                        loss = loss * same_rot_idx
                    ## update the final prob-adjusted losses
                    final_losses[name] = loss.mean()

                logit_loss_target = collapseBF(logit_loss_target).detach()  # detach the gradient for the loss target
                final_losses['logit_loss'] = ((rot_logit - logit_loss_target) ** 2.).mean()
                final_losses['logit_loss_target'] = logit_loss_target.mean()

        ## regularizers
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                regularizers, aux = self.compute_regularizers(
                    arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                    posed_bones=forward_aux.get("posed_bones"), prior_shape=prior_shape, mvp=mvp,
                    mask_gt=mask_gt, keypoint_gt=keypoint_gt, keypoint_gt_flag=keypoint_gt_flag
                )
        else:
            regularizers, aux = self.compute_regularizers(
                arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                posed_bones=forward_aux.get("posed_bones"), prior_shape=prior_shape, mvp=mvp,
                mask_gt=mask_gt, keypoint_gt=keypoint_gt, keypoint_gt_flag=keypoint_gt_flag
            )
        final_losses.update(regularizers)
        aux_viz.update(aux)

        ## compute final losses
        total_loss = 0
        for name, loss in final_losses.items():
            loss_weight = getattr(self.cfg_loss, f"{name}_weight")
            if loss_weight <= 0:
                continue
            if not in_range(total_iter, self.cfg_predictor_instance.cfg_texture.texture_iter_range) and (
                    name in ['rgb_loss']):
                continue
            if not in_range(total_iter, self.cfg_loss.arti_reg_loss_iter_range) and (name in ['arti_reg_loss']):
                continue
            if name in ["logit_loss_target"]:
                continue
            total_loss += loss * loss_weight
        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import pdb;
            pdb.set_trace()

        metrics = {'loss': total_loss, **final_losses}

        log = SimpleNamespace(**locals())
        if self.accelerator.is_main_process and logger is not None and (self.enable_render or not is_training):
            self.log_visuals(log, logger)
        if self.accelerator.is_main_process and save_results:
            self.save_results(log)
        return metrics

    @torch.no_grad()
    def log_visuals(self, log, logger, sdf_feats=None, text=None):
        # Add keypoint visualization to image_gt
        if log.keypoint_gt is not None:
            log.input_image = draw_keypoints(log.input_image, log.keypoint_gt, log.keypoint_gt_flag)
        return super().log_visuals(log, logger, sdf_feats, text)


