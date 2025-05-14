import cv2
import torch
from einops import rearrange, repeat
import torch.nn.functional as F
from externals.Animals.model.models.Fauna import FaunaModel
from externals.Animals.model.utils import misc



class FaunaInference(FaunaModel):
    def __init__(self, config):
        super().__init__(config)

    # def inference(self, batch, epoch=200000, total_iter=20000000, is_training=False, save_dir=None):
    #     input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, depth_gt, seq_idx, frame_idx, paths,= batch
    #     if bbox.shape[2] == 9:
    #         # Fauna Dataset bbox
    #         global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, tmp_label = bbox.unbind(
    #             2)  # BxFx9
    #     elif bbox.shape[2] == 8:
    #         # in visualization using magicpony dataset for simplicity
    #         global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
    #     else:
    #         raise NotImplementedError
    #
    #     mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
    #     mask_dt = mask_dt / self.dataset.in_image_size
    #     batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
    #     h = w = self.dataset.out_image_size
    #     aux_viz = {}
    #
    #     dino_feat_im_gt = None if dino_feat_im is None else expandBF(
    #         torch.nn.functional.interpolate(collapseBF(dino_feat_im), size=[h, w], mode="bilinear"), batch_size,
    #         num_frames)[:, :, :self.cfg_predictor_base.cfg_dino.feature_dim]
    #     dino_cluster_im_gt = None if dino_cluster_im is None else expandBF(
    #         torch.nn.functional.interpolate(collapseBF(dino_cluster_im), size=[h, w], mode="nearest"), batch_size,
    #         num_frames)
    #
    #     ## GT image
    #     image_gt = input_image
    #     if self.dataset.out_image_size != self.dataset.in_image_size:
    #         image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'),
    #                             batch_size, num_frames)
    #         if flow_gt is not None:
    #             flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"),
    #                                batch_size, num_frames - 1)
    #
    #     ## predict prior shape and DINO
    #     if in_range(total_iter, self.cfg_predictor_base.cfg_shape.grid_res_coarse_iter_range):
    #         grid_res = self.cfg_predictor_base.cfg_shape.grid_res_coarse
    #     else:
    #         grid_res = self.cfg_predictor_base.cfg_shape.grid_res
    #     if self.get_predictor("netBase").netShape.grid_res != grid_res:
    #         self.get_predictor("netBase").netShape.load_tets(grid_res)
    #     if self.mixed_precision:
    #         with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
    #             prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
    #                                                                  batch=batch, bank_enc=self.get_predictor(
    #                     "netInstance").netEncoder)
    #     else:
    #         prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
    #                                                              batch=batch,
    #                                                              bank_enc=self.get_predictor("netInstance").netEncoder)
    #
    #     class_vector = bank_embedding[0]
    #
    #     ## predict instance specific parameters
    #     if self.mixed_precision:
    #         with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
    #             shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
    #                 input_image, prior_shape, epoch, total_iter, is_training=is_training)
    #         pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
    #             map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
    #     else:
    #         shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
    #             input_image, prior_shape, epoch, total_iter,
    #             is_training=is_training)  # first two dim dimensions already collapsed N=(B*F)
    #     # if not is_training and (batch_size != arti_params.shape[0] or num_frames != arti_params.shape[1]):
    #     #     # If b f sampled from vae different from training b f
    #     #     batch_size, num_frames = arti_params.shape[:2]
    #     rot_logit = forward_aux['rot_logit']
    #     rot_idx = forward_aux['rot_idx']
    #     rot_prob = forward_aux['rot_prob']
    #     aux_viz.update(forward_aux)
    #     final_losses = {}
    #
    #     ## render images
    #     if self.enable_render or not is_training:  # Force render for val and test
    #         render_flow = self.cfg_render.render_flow and num_frames > 1
    #         render_modes = ['shaded', 'dino_pred']
    #         if render_flow:
    #             render_modes += ['flow']
    #         if self.mixed_precision:
    #             with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
    #                 renders = self.render(
    #                     render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
    #                     prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames,
    #                     class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
    #                 )
    #         else:
    #             renders = self.render(
    #                 render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
    #                 prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames,
    #                 class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
    #             )
    #         renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
    #         if render_flow:
    #             shaded, dino_feat_im_pred, flow_pred = renders
    #             flow_pred = flow_pred[:, :-1]  # Bx(F-1)x2xHxW
    #         else:
    #             shaded, dino_feat_im_pred = renders
    #             flow_pred = None
    #         image_pred = shaded[:, :, :3]
    #         mask_pred = shaded[:, :, 3]
    #     b0 = batch_size * num_frames
    #     misc.save_obj(
    #         out_fold=os.path.dirname(paths[0][0]), fnames=[os.path.basename(p).format("mesh") for p in paths[0]],
    #         meshes=shape.first_n(b0), feat=im_features[:b0], save_material=False
    #     )
    #     return shape

    def forward(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='',
                is_training=True):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, keypoint_gt, seq_idx, frame_idx = batch
        if bbox.shape[2] == 9:
            # Fauna Dataset bbox
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness, tmp_label = bbox.unbind(
                2)  # BxFx9
        elif bbox.shape[2] == 8:
            # in visualization using magicpony dataset for simplicity
            global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        else:
            raise NotImplementedError

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
        if self.dataset.out_image_size != self.dataset.in_image_size:
            image_gt = expandBF(torch.nn.functional.interpolate(collapseBF(image_gt), size=[h, w], mode='bilinear'),
                                batch_size, num_frames)
            if flow_gt is not None:
                flow_gt = expandBF(torch.nn.functional.interpolate(collapseBF(flow_gt), size=[h, w], mode="bilinear"),
                                   batch_size, num_frames - 1)

        ## predict prior shape and DINO
        if in_range(total_iter, self.cfg_predictor_base.cfg_shape.grid_res_coarse_iter_range):
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res_coarse
        else:
            grid_res = self.cfg_predictor_base.cfg_shape.grid_res
        if self.get_predictor("netBase").netShape.grid_res != grid_res:
            self.get_predictor("netBase").netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
                                                                     batch=batch, bank_enc=self.get_predictor(
                        "netInstance").netEncoder)
        else:
            prior_shape, dino_net, bank_embedding = self.netBase(total_iter=total_iter, is_training=is_training,
                                                                 batch=batch,
                                                                 bank_enc=self.get_predictor("netInstance").netEncoder)

        class_vector = bank_embedding[0]

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                    input_image, prior_shape, epoch, total_iter, is_training=is_training)
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                input_image, prior_shape, epoch, total_iter,
                is_training=is_training)  # first two dim dimensions already collapsed N=(B*F)
        # if not is_training and (batch_size != arti_params.shape[0] or num_frames != arti_params.shape[1]):
        #     # If b f sampled from vae different from training b f
        #     batch_size, num_frames = arti_params.shape[:2]
        keypoint_gt_flag = disable_keypoint_loss(w2c, b=batch_size)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)
        final_losses = {}

        ## render images
        if self.enable_render or not is_training:  # Force render for val and test
            render_flow = self.cfg_render.render_flow and num_frames > 1
            render_modes = ['shaded', 'dino_pred']
            if render_flow:
                render_modes += ['flow']
            if self.mixed_precision:
                with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                    renders = self.render(
                        render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                        prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames,
                        class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                    )
            else:
                renders = self.render(
                    render_modes, shape, texture, mvp, w2c, campos, (h, w), im_features=im_features, light=light,
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames,
                    class_vector=class_vector[None, :].expand(batch_size * num_frames, -1)
                )
            renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
            if render_flow:
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
                        dino_feat_im_gt,
                        dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False
                    )
            else:
                losses = self.compute_reconstruction_losses(
                    image_pred, image_gt, mask_pred, mask_gt, mask_dt, mask_valid, flow_pred, flow_gt, dino_feat_im_gt,
                    dino_feat_im_pred, background_mode=self.cfg_render.background_mode, reduce=False
                )

            ## supervise the rotation logits directly with reconstruction loss
            logit_loss_target = None
            if losses is not None:
                logit_loss_target = torch.zeros_like(expandBF(rot_logit, batch_size, num_frames))
                for name, loss in losses.items():
                    loss_weight = getattr(self.cfg_loss, f"{name}_weight")
                    if name in ['dino_feat_im_loss']:
                        ## increase the importance of dino loss for viewpoint hypothesis selection (directly increasing dino recon loss leads to stripe artifacts)
                        # loss_weight = loss_weight * self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier
                        loss_weight = self.parse_dict_definition(self.cfg_loss.dino_feat_im_loss_weight_dict,
                                                                 total_iter)
                        loss_weight = loss_weight * self.parse_dict_definition(
                            self.cfg_loss.logit_loss_dino_feat_im_loss_multiplier_dict, total_iter)
                    if name in ['mask_loss']:
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_mask_multiplier
                    if name in ['mask_inv_dt_loss']:
                        loss_weight = loss_weight * self.cfg_loss.logit_loss_mask_inv_dt_multiplier
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

        random_view_aux = None
        random_view_aux = self.get_random_view_mask(w2c, shape, prior_shape, num_frames)
        if (self.cfg_mask_discriminator.enable_iter[0] < total_iter) and (
                self.cfg_mask_discriminator.enable_iter[1] > total_iter):
            disc_loss = self.compute_mask_disc_loss_gen(
                mask_gt, mask_pred, random_view_aux['mask_random_pred'], condition_feat=class_vector
            )
            final_losses.update(disc_loss)

        ## regularizers
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                regularizers, aux = self.compute_regularizers(
                    arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                    posed_bones=forward_aux.get("posed_bones"), keypoint_gt=keypoint_gt, keypoint_gt_flag=keypoint_gt_flag,
                    class_vector=class_vector.detach() if class_vector is not None else None
                )
        else:
            regularizers, aux = self.compute_regularizers(
                arti_params=arti_params, deformation=deformation, pose_raw=pose_raw,
                posed_bones=forward_aux.get("posed_bones"), keypoint_gt=keypoint_gt, keypoint_gt_flag=keypoint_gt_flag,
                class_vector=class_vector.detach() if class_vector is not None else None
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

            if name == 'dino_feat_im_loss':
                loss_weight = self.parse_dict_definition(self.cfg_loss.dino_feat_im_loss_weight_dict, total_iter)

            total_loss += loss * loss_weight
        self.total_loss += total_loss  # reset to 0 in backward step

        if torch.isnan(self.total_loss):
            print("NaN in loss...")
            import pdb;
            pdb.set_trace()

        metrics = {'loss': total_loss, **final_losses}

        log = SimpleNamespace(**locals())
        if logger is not None and (self.enable_render or not is_training):
            self.log_visuals(log, logger)
        if save_results:
            self.save_results(log)
        return metrics

    def compute_regularizers(
        self, arti_params=None, deformation=None, pose_raw=None, posed_bones=None, class_vector=None, prior_shape=None,
        **kwargs
    ):
        losses = {}
        aux = {}
        losses.update(self.get_predictor("netBase").netShape.get_sdf_reg_loss(feats=class_vector))
        if arti_params is not None:
            losses['arti_reg_loss'] = (arti_params ** 2).mean()
        if deformation is not None:
            losses['deform_reg_loss'] = (deformation ** 2).mean()

        # Smooth losses
        if self.dataset.data_type == "sequence" and self.dataset.num_frames > 1:
            b, f = self.dataset.batch_size, self.dataset.num_frames
            if self.cfg_loss.deform_smooth_loss_weight > 0 and deformation is not None:
                losses["deform_smooth_loss"] = self.smooth_loss_fn(expandBF(deformation, b, f))
            if arti_params is not None:
                if self.cfg_loss.arti_smooth_loss_weight > 0:
                    losses["arti_smooth_loss"] = self.smooth_loss_fn(arti_params)
                if self.cfg_loss.artivel_smooth_loss_weight > 0:
                    artivel = arti_params[:, 1:, ...] - arti_params[:, :(f-1), ...]
                    losses["artivel_smooth_loss"] = self.smooth_loss_fn(artivel)
            if pose_raw is not None:
                campose = expandBF(pose_raw, b, f)
                if self.cfg_loss.campose_smooth_loss_weight > 0:
                    losses["campose_smooth_loss"] = self.smooth_loss_fn(campose)
                if self.cfg_loss.camposevel_smooth_loss_weight > 0:
                    camposevel = campose[:, 1:, ...] - campose[:, :(f-1), ...]
                    losses["camposevel_smooth_loss"] = self.smooth_loss_fn(camposevel)
            if posed_bones is not None:
                if self.cfg_loss.bone_smooth_loss_weight > 0:
                    losses["bone_smooth_loss"] = self.smooth_loss_fn(posed_bones)
                if self.cfg_loss.bonevel_smooth_loss_weight > 0:
                    bonevel = posed_bones[:, 1:, ...] - posed_bones[:, :(f-1), ...]
                    losses["bonevel_smooth_loss"] = self.smooth_loss_fn(bonevel)
        return losses, aux


def in_range(x, range):
    return misc.in_range(x, range, default_indicator=-1)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')


def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def get_optimizer(model, lr=0.0001, betas=(0.9, 0.999), weight_decay=0, eps=1e-8):
    return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)

def to_float(x):
    try:
        return x.float()
    except AttributeError:
        return x


def disable_keypoint_loss(w2c, b):
    object_front_normal = repeat(torch.tensor([1., 0., 0.], device=w2c.device), "c -> b c 1", b=b)
    R_world_to_cam = w2c[:, :3, :3]
    cam_forward_in_world = R_world_to_cam.transpose(1, 2) @ repeat(
        torch.tensor([0., 0., 1.], device=w2c.device), "c -> b c 1", b=b
    )
    similarity = F.cosine_similarity(cam_forward_in_world, object_front_normal).abs()
    keypoint_gt_flag = (similarity > 0.5).squeeze()
    return keypoint_gt_flag


def draw_keypoints(image_t, keypoint, gt_flag, circle_color=(0, 0, 255), text_color=(0, 255, 255), radius=3):
    b, f = image_t.shape[:2]
    img_size = image_t.shape[-1]
    image_t = rearrange(image_t, "b f ... -> (b f) ...")
    keypoint = rearrange((keypoint + 1) / 2 * img_size, "b f ... -> (b f) ...")
    out_list = []
    for img, k, flag in zip(image_t, keypoint, gt_flag):
        if flag:
            img_np = (img.clone() * 255).permute(1, 2, 0).clamp(0, 255).contiguous().cpu().numpy().astype(np.uint8)
            for i, (x, y) in enumerate(k[:, :2]):
                x, y = int(x.item()), int(y.item())
                cv2.circle(img_np, center=(x, y), radius=radius, color=circle_color, thickness=-1)
                cv2.putText(
                    img_np, text=str(i), org=(x + radius + 1, y - radius - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=text_color, thickness=1, lineType=cv2.LINE_AA
                )
            out_list.append(torch.from_numpy(img_np).permute(2, 0, 1) / 255)
        else:
            out_list.append(img.cpu())
    out_t = rearrange(torch.stack(out_list), "(b f) ... -> b f ...", b=b, f=f)
    return out_t
