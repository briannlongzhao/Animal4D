from dataclasses import dataclass
import torch
import random
import os
import numpy as np
from itertools import chain
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from externals.Animals.model.utils import misc
from externals.Animals.model.geometry.skinning import skinning
from externals.Animals.model.render import mesh
from externals.Animals.model.models.AnimalModel import AnimalModelConfig, AnimalModel
from externals.Animals.model.predictors import BasePredictorBase, BasePredictorConfig, in_range
from externals.Animals.model.predictors.InstancePredictorBase import InstancePredictorBase, InstancePredictorConfig
from externals.Animals.model.geometry.skinning import euler_angles_to_matrix

from models.animal_models.models import MagicponyFinetune
from models.animal_models.utils import *


class InstancePredictorMagicponyVideo(InstancePredictorBase):
    def __init__(self, cfg: InstancePredictorConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, InstancePredictorConfig)
        self.articulation_dict = TensorDict()
        self.pose_dict = TensorDict()

    def forward_deformation(self, shape, feat=None, batch_size=None, num_frames=None):
        original_verts = shape.v_pos
        num_verts = original_verts.shape[1]
        deform_feat = None
        if feat is not None:
            deform_feat = feat[:, None, :].repeat(1, num_verts, 1)  # Shape: (B, num_verts, latent_dim)
            original_verts = original_verts.repeat(len(feat), 1, 1)
        deformation = self.netDeform(original_verts, deform_feat) * 0.1  # Shape: (B, num_verts, 3), multiply by 0.1 to minimize disruption when initially enabled
        if deformation.shape[0] > 1 and self.cfg_deform.force_avg_deform:
            assert batch_size is not None and num_frames is not None
            assert deformation.shape[0] == batch_size * num_frames
            deformation = deformation.view(batch_size, num_frames, *deformation.shape[1:])
            deformation = deformation.mean(dim=1, keepdim=True)
            deformation = deformation.repeat(1,num_frames,*[1]*(deformation.dim()-2))
            deformation = deformation.view(batch_size*num_frames, *deformation.shape[2:])
        shape = shape.deform(deformation)
        return shape, deformation

    def forward_pose(self, patch_out, patch_key, frame_ids=None, **kwargs):
        # Add netPose prediction to articulation_dict if not in dict
        with torch.no_grad():
            if self.cfg_pose.architecture == 'encoder_dino_patch_key':
                pose_gt = self.netPose(patch_key)  # Shape: (B, latent_dim)
            elif self.cfg_pose.architecture == 'encoder_dino_patch_out':
                pose_gt = self.netPose(patch_out)  # Shape: (B, latent_dim)
            else:
                raise NotImplementedError
        frame_ids = rearrange(frame_ids, "b f -> (b f)")
        self.pose_dict[frame_ids] = pose_gt  # no-op if frame_ids already in dict
        pose = self.pose_dict(frame_ids)

        ## xyz translation
        trans_pred = pose[..., -3:].tanh() * self.max_trans_xyz_range.to(pose.device)

        ## rotation
        if self.cfg_pose.rot_rep == 'euler_angle':
            rot_pred = pose[..., :3].tanh() * self.max_rot_xyz_range.to(pose.device)

        elif self.cfg_pose.rot_rep == 'quaternion':
            quat_init = torch.FloatTensor([0.01, 0, 0, 0]).to(pose.device)
            rot_pred = pose[..., :4] + quat_init
            rot_pred = torch.nn.functional.normalize(rot_pred, p=2, dim=-1)
            # rot_pred = torch.cat([rot_pred[...,:1].abs(), rot_pred[...,1:]], -1)  # make real part non-negative
            rot_pred = rot_pred * rot_pred[..., :1].sign()  # make real part non-negative

        elif self.cfg_pose.rot_rep == 'lookat':
            vec_forward = pose[..., :3]
            if self.cfg_pose.lookat_zeroy:
                vec_forward = vec_forward * torch.FloatTensor([1, 0, 1]).to(pose.device)
            vec_forward = torch.nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = vec_forward

        elif self.cfg_pose.rot_rep in ['quadlookat', 'octlookat']:
            rots_pred = pose[..., :self.num_pose_hypos * 4].view(-1, self.num_pose_hypos, 4)  # (B*F, K, 4)
            rots_logits = rots_pred[..., :1]
            vec_forward = rots_pred[..., 1:4]

            def softplus_with_init(x, init=0.5):
                assert np.abs(init) > 1e-8, "initial value should be non-zero"
                beta = np.log(2) / init
                return torch.nn.functional.softplus(x, beta=beta)

            xs, ys, zs = vec_forward.unbind(-1)
            xs = softplus_with_init(xs, init=0.5)  # initialize to 0.5
            if self.cfg_pose.rot_rep == 'octlookat':
                ys = softplus_with_init(ys, init=0.5)  # initialize to 0.5
            if self.cfg_pose.lookat_zeroy:
                ys = ys * 0
            zs = softplus_with_init(zs, init=0.5)  # initialize to 0.5
            vec_forward = torch.stack([xs, ys, zs], -1)
            vec_forward = vec_forward * self.orthant_signs.to(pose.device)
            vec_forward = torch.nn.functional.normalize(vec_forward, p=2, dim=-1)  # x right, y up, z forward
            rot_pred = torch.cat([rots_logits, vec_forward], -1).view(-1, self.num_pose_hypos * 4)  # (B*F, K*4)

        else:
            raise NotImplementedError

        pose = torch.cat([rot_pred, trans_pred], -1)
        return pose

    def forward_articulation(self, shape, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter, frame_ids=None, **kwargs):
        """
        Inherited from InstancePredictorBase, additionally take frame_ids for arti params reconstruction
        """
        verts = shape.v_pos
        if len(verts) == batch_size * num_frames:
            verts = verts.view(batch_size, num_frames, *verts.shape[1:])  # BxFxNx3
        else:
            verts = verts[None]  # 1x1xNx3

        bones, bones_feat, bones_pos_in = self.get_bones(
            verts, feat, patch_feat, mvp, w2c, batch_size, num_frames, epoch, total_iter
        )

        # forward motion reconstruction using frame_ids get pred articulation angles
        b, f = frame_ids.shape
        frame_ids = rearrange(frame_ids, "b f -> (b f)")
        # Add netArticulation prediction to articulation_dict if not in dict
        with torch.no_grad():
            articulation_angles_out = self.netArticulation(bones_feat, bones_pos_in).view(batch_size, num_frames, bones.shape[2], 3)
            self.articulation_angles_gt = self.apply_articulation_constraints(articulation_angles_out.clone().detach())
        articulation_angles_out = rearrange(articulation_angles_out, "b f n c -> (b f) n c", b=b, f=f)
        self.articulation_dict[frame_ids] = articulation_angles_out  # no-op if frame_ids already in dict
        articulation_angles_pred = self.articulation_dict(frame_ids)
        articulation_angles_pred = rearrange(articulation_angles_pred, "(b f) n c -> b f n c", b=b, f=f)
        articulation_angles_pred = self.apply_articulation_constraints(articulation_angles_pred)
        self.articulation_angles_pred = articulation_angles_pred

        # skinning and make pred shape
        verts_articulated_pred, aux = skinning(
            verts, bones, self.kinematic_tree, articulation_angles_pred, output_posed_bones=True,
            temperature=self.cfg_articulation.skinning_temperature
        )
        verts_articulated_pred = verts_articulated_pred.view(batch_size * num_frames, *verts_articulated_pred.shape[2:])
        v_tex = shape.v_tex
        if len(v_tex) != len(verts_articulated_pred):
            v_tex = v_tex.repeat(len(verts_articulated_pred), 1, 1)
        articulated_shape_pred = mesh.make_mesh(
            verts_articulated_pred, shape.t_pos_idx, v_tex, shape.t_tex_idx, shape.material
        )
        return articulated_shape_pred, articulation_angles_pred, aux



class MagicponyVideo(MagicponyFinetune):
    def __init__(self, cfg: AnimalModelConfig):
        super().__init__(cfg)
        misc.load_cfg(self, cfg, AnimalModelConfig)
        self.netInstance = InstancePredictorMagicponyVideo(self.cfg_predictor_instance)

    def set_finetune_arti(self):
        super().set_train()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters()):
            param.requires_grad = False
        for param in chain(self.netInstance.articulation_dict.parameters(), self.netInstance.pose_dict.parameters()):
            param.requires_grad = True
        # Set a learnable fov
        self.netInstance.fov = torch.nn.Parameter(torch.tensor([self.netInstance.cfg_pose.fov], dtype=torch.float32))
        self.netInstance.get_camera_extrinsics_from_pose = get_camera_extrinsics_from_pose_differentiable.__get__(self.netInstance, InstancePredictorBase)
        self.netInstance.enable_deform = False
        self.optimizerInstance.add_param_group({"params": [self.netInstance.fov]})

    def set_finetune_texture(self):
        super().set_train()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters()):
            param.requires_grad = False
        for param in chain(self.netInstance.netTexture.parameters()):
            param.requires_grad = True
        self.netInstance.enable_deform = True

    def set_inference(self):
        super().set_eval()
        for param in chain(self.netInstance.parameters(), self.netBase.parameters()):
            param.requires_grad = False


    # def get_default_pose(self):
    #     pose_canon = torch.concat([torch.eye(3), torch.zeros(1, 3)], dim=0).view(-1)[None].to(self.device)
    #     mvp_canon, w2c_canon, campos_canon = self.netInstance.get_camera_extrinsics_from_pose(pose_canon, offset_extra=self.cfg_render.offset_extra)
    #     viewpoint_arti = torch.FloatTensor([0, -120, 0]) / 180 * np.pi
    #     mtx = torch.eye(4).to(self.device)
    #     mtx[:3, :3] = euler_angles_to_matrix(viewpoint_arti, "XYZ")
    #     w2c_arti = torch.matmul(w2c_canon, mtx[None])
    #     mvp_arti = torch.matmul(mvp_canon, mtx[None])
    #     campos_arti = campos_canon @ torch.linalg.inv(mtx[:3, :3]).T
    #     self.default_pose, self.default_mvp, self.default_w2c, self.default_campos = pose_canon, mvp_arti, w2c_arti, campos_arti


    @torch.no_grad()
    def log_visuals(self, log, logger):
        log = super().log_visuals(log, logger)
        b0 = max(min(log.batch_size, 16 // log.num_frames), 1)
        def log_image(name, image):
            logger.add_image(log.logger_prefix + 'image/' + name, misc.image_grid(collapseBF(image[:b0, :]).detach().cpu().clamp(0, 1)), log.total_iter)
        def log_video(name, frames, fps=5):
            logger.add_video(log.logger_prefix+'animation/'+name, frames.detach().cpu().unsqueeze(0).clamp(0,1), log.total_iter, fps=fps)
        if log.num_frames > 1:
            log_video("sequence_image_gt", log.input_image[0])
            log_video("sequence_mask_gt", repeat(log.mask_gt[0], "f h w -> f c h w", c=3))
            suffix = "pred"
            log_video(f"sequence_image_{suffix}", log.image_pred[0])
            log_video(f"sequence_mask_{suffix}", repeat(log.mask_pred[0], "f h w -> f c h w", c=3))
            log_video(f"sequence_instance_geo_normal_{suffix}", log.geo_normal[0])
            if hasattr(log, "geo_normal_gt"):
                log_video(f"sequence_instance_geo_normal_gt", log.geo_normal_gt[0])

            leg_bones = [8, 11, 14, 17]
            all_leg_bone_pos = []
            for frame_ids, posed_bones in zip(log.global_frame_id, log.aux_viz['posed_bones']):  # Iterate batch
                frame_id_to_posed_bones = {}
                for frame_id, posed_bones in zip(frame_ids, posed_bones):  # Get unique frames
                    frame_id_to_posed_bones[int(frame_id.item())] = posed_bones  # (20,2,3)
                fig, ax = plt.subplots(figsize=(5, 5))
                frame_ids = sorted(frame_id_to_posed_bones.keys())
                for bone_idx in leg_bones:
                    motion = [frame_id_to_posed_bones[frame_id][bone_idx, 0, 2].item() for frame_id in frame_ids]
                    ax.plot(frame_ids, motion, 'o-', label=bone_idx)
                ax.legend()
                fig.canvas.draw()
                img_str = fig.canvas.tostring_rgb()
                width, height = fig.canvas.get_width_height()
                plt.close(fig)
                img = np.frombuffer(img_str, dtype=np.uint8).reshape(height, width, 3) / 255.
                all_leg_bone_pos.append(rearrange(img, "h w c -> 1 c h w"))
            all_leg_bone_pos = torch.FloatTensor(np.stack(all_leg_bone_pos))
            log_image(f"leg_bone_pos", all_leg_bone_pos)

    def forward_finetune_arti(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True, **kwargs):
        batch = batch[:-1]  # exclude path
        m = super().forward(batch, epoch, logger, total_iter, save_results, save_dir, logger_prefix, is_training)
        # Add newly added per frame params to optimizer
        for name in ["articulation", "pose"]:
            existing_frames = [
                d.get("frame_idx") for d in self.optimizerInstance.param_groups
                if d.get("frame_idx") is not None and d.get("name") == name]
            param_dict = getattr(self.netInstance, f"{name}_dict")
            for frame_idx, param in param_dict.items():
                if int(frame_idx) not in existing_frames:
                    self.optimizerInstance.add_param_group(
                        {"params": [param], "name": name, "frame_idx": int(frame_idx)}
                    )
            print(self.netInstance.fov.item())
        return m

    def forward_finetune_texture(self, batch, epoch, logger=None, total_iter=None, save_results=False, save_dir=None, logger_prefix='', is_training=True, **kwargs):
        batch = batch[:-1]
        m = super().forward(batch, epoch, logger, total_iter, save_results, save_dir, logger_prefix, is_training)
        return m

    @torch.no_grad()
    def inference(self, batch, total_iter, epoch, local_save_dir=None,):
        input_image, mask_gt, mask_dt, mask_valid, flow_gt, bbox, bg_image, dino_feat_im, dino_cluster_im, keypoint, seq_idx, frame_idx, paths = batch
        global_frame_id, crop_x0, crop_y0, crop_w, crop_h, full_w, full_h, sharpness = bbox.unbind(2)  # BxFx8
        mask_gt = (mask_gt[:, :, 0, :, :] > 0.9).float()  # BxFxHxW
        mask_dt = mask_dt / self.dataset.in_image_size
        batch_size, num_frames, _, _, _ = input_image.shape  # BxFxCxHxW
        h = w = self.dataset.out_image_size
        aux_viz = {}

        ## predict prior shape and DINO
        grid_res = self.cfg_predictor_base.cfg_shape.grid_res
        if self.get_predictor("netBase").netShape.grid_res != grid_res:
            self.get_predictor("netBase").netShape.load_tets(grid_res)
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                prior_shape, dino_net = self.netBase(total_iter=total_iter, is_training=False)
        else:
            prior_shape, dino_net = self.netBase(total_iter=total_iter, is_training=False)

        ## predict instance specific parameters
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                    input_image, prior_shape, epoch, total_iter, is_training=False, frame_ids=global_frame_id)  # first two dim dimensions already collapsed N=(B*F
            pose_raw, pose, mvp, w2c, campos, im_features, arti_params = \
                map(to_float, [pose_raw, pose, mvp, w2c, campos, im_features, arti_params])
        else:
            shape, pose_raw, pose, mvp, w2c, campos, texture, im_features, deformation, arti_params, light, forward_aux = self.netInstance(
                input_image, prior_shape, epoch, total_iter, is_training=False, frame_ids=global_frame_id,)  # first two dim dimensions already collapsed N=(B*F)
        rot_logit = forward_aux['rot_logit']
        rot_idx = forward_aux['rot_idx']
        rot_prob = forward_aux['rot_prob']
        aux_viz.update(forward_aux)
        final_losses = {}

        ## render images
        render_flow = self.cfg_render.render_flow and num_frames > 1
        render_modes = ['geo_normal', 'shaded', 'dino_pred']
        if render_flow:
            render_modes += ['flow']
        if self.mixed_precision:
            with torch.autocast(device_type=torch.device(self.accelerator.device).type, dtype=self.mixed_precision):
                renders = self.render(
                    render_modes, shape, texture, mvp, w2c, campos, (h, w),
                    im_features=im_features, light=light,
                    prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
                )
        else:
            renders = self.render(
                render_modes, shape, texture, mvp, w2c, campos, (h, w),
                im_features=im_features, light=light,
                prior_shape=prior_shape, dino_net=dino_net, num_frames=num_frames
            )
        b0 = batch_size * num_frames
        if b0 != renders[0].shape[0]:
            batch_size = int(renders[0].shape[0] / num_frames)
        renders = map(lambda x: expandBF(x, batch_size, num_frames), renders)
        geo_normal, shaded, dino_feat_im_pred = renders
        image_pred = shaded[:, :, :3]
        paths = [item for sublist in paths for item in sublist]
        if not os.path.isdir(local_save_dir):
            os.makedirs(local_save_dir)
        misc.save_images(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format("normal") for p in paths],
            imgs=geo_normal.squeeze(0).detach().cpu().numpy()
        )
        misc.save_obj(
            out_fold=local_save_dir, fnames=[os.path.basename(p).format("mesh") for p in paths],
            meshes=shape.first_n(b0), feat=im_features[:b0], save_material=False
        )
        return



def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')


def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


# Differentialble with respect to fovy
def get_camera_extrinsics_from_pose_differentiable(self, pose, znear=0.1, zfar=1000., offset_extra=None):
    def perspective(fovy, aspect=1.0, n=0.1, f=1000.0, device=None):
        y = torch.tan(fovy / 2)
        mat = torch.zeros((4, 4), dtype=torch.float32, device=device)
        mat[0, 0] = 1.0 / (y * aspect)
        mat[1, 1] = 1.0 / (-y)
        mat[2, 2] = -(f + n) / (f - n)
        mat[2, 3] = -(2 * f * n) / (f - n)
        mat[3, 2] = -1.0
        return mat
    N = len(pose)
    pose_R = pose[:, :9].view(N, 3, 3).transpose(2, 1)  # to be compatible with pytorch3d
    if offset_extra is not None:
        cam_pos_offset = torch.FloatTensor([0, 0, -self.cfg_pose.cam_pos_z_offset - offset_extra]).to(pose.device)
    else:
        cam_pos_offset = torch.FloatTensor([0, 0, -self.cfg_pose.cam_pos_z_offset]).to(pose.device)
    pose_T = pose[:, -3:] + cam_pos_offset[None, None, :]
    pose_T = pose_T.view(N, 3, 1)
    pose_RT = torch.cat([pose_R, pose_T], axis=2)  # Nx3x4
    w2c = torch.cat([pose_RT, torch.FloatTensor([0, 0, 0, 1]).repeat(N, 1, 1).to(pose.device)], axis=1)  # Nx4x4
    proj = perspective(self.fov / 180 * np.pi, 1, znear, zfar)[None].to(pose.device)  # assuming square images
    mvp = torch.matmul(proj, w2c)
    campos = -torch.matmul(pose_R.transpose(2, 1), pose_T).view(N, 3)
    return mvp, w2c, campos

