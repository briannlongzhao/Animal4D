import os
import cv2
import numpy as np
from glob import glob
from einops import rearrange, repeat
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from nvdiffrast.torch import RasterizeGLContext, rasterize

from externals.Animals.model.utils import misc
from externals.Animals.model.render.renderutils import xfm_points




def in_range(x, range):
    return misc.in_range(x, range, default_indicator=-1)


def collapseBF(x):
    return None if x is None else rearrange(x, 'b f ... -> (b f) ...')


def expandBF(x, b, f):
    return None if x is None else rearrange(x, '(b f) ... -> b f ...', b=b, f=f)


def to_float(x):
    try:
        return x.float()
    except AttributeError:
        return x


def disable_keypoint_loss(w2c, b):
    object_front_normal = repeat(torch.tensor([1., 0., 0.], device=w2c.device), "c -> b c 1", b=b)
    R_world_to_cam = w2c[:, :3, :3]
    cam_forward_in_world = R_world_to_cam.transpose(1, 2) @ repeat(torch.tensor([0., 0., 1.], device=w2c.device),
                                                                   "c -> b c 1", b=b)
    similarity = F.cosine_similarity(cam_forward_in_world, object_front_normal).abs()
    keypoint_gt_flag = (similarity > 0.25).squeeze()
    return keypoint_gt_flag


def draw_keypoints(image_t, keypoint, gt_flag=None, circle_color=(0, 0, 255), text_color=(0, 255, 255), radius=3):
    b, f = image_t.shape[:2]
    img_size = image_t.shape[-1]
    image_t = rearrange(image_t, "b f ... -> (b f) ...")
    keypoint = rearrange((keypoint + 1) / 2 * img_size, "b f ... -> (b f) ...")
    if gt_flag is None:
        gt_flag = torch.ones(b*f, device=image_t.device, dtype=torch.bool)
    elif gt_flag.dim() == 2:
        gt_flag = rearrange(gt_flag, "b f -> (b f)")
    out_list = []
    for img, k, flag in zip(image_t, keypoint, gt_flag):
        if flag:
            img_np = (img.clone() * 255).permute(1, 2, 0).clamp(0, 255).contiguous().cpu().numpy().astype(np.uint8)
            for i, (x, y) in enumerate(k[:, :2]):
                x, y = int(x.item()), int(y.item())
                cv2.circle(img_np, center=(x, y), radius=radius, color=circle_color, thickness=-1)
                cv2.putText(
                    img_np, text=str(i), org=(x + radius + 1, y - radius - 1),  fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=text_color, thickness=1, lineType=cv2.LINE_AA
                )
            out_list.append(torch.from_numpy(img_np).permute(2, 0, 1) / 255)
        else:
            out_list.append(img.cpu())
    out_t = rearrange(torch.stack(out_list), "(b f) ... -> b f ...", b=b, f=f)
    return out_t


def save_checkpoint(checkpoint_dir, model, epoch=0, total_iter=0, save_optim=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint{total_iter}.pth')
    state_dict = model.get_model_state()
    if save_optim:
        optimizer_state = model.get_optimizer_state()
        state_dict = {**state_dict, **optimizer_state}
    state_dict['epoch'] = epoch
    state_dict['total_iter'] = total_iter
    print(f"Saving checkpoint to {checkpoint_path}")
    torch.save(state_dict, checkpoint_path)



def finetune(
    model, train_dataloader, test_dataloader, logger,
    total_iter, total_epoch, finetune_iter, log_image_freq, log_loss_freq, device
):
    test_iter = iter(test_dataloader)
    max_iter = total_iter + finetune_iter
    while total_iter < max_iter:
        for batch in train_dataloader:
            total_iter += 1
            log_image = total_iter % log_image_freq == 0
            log_loss = total_iter % log_loss_freq == 0
            batch = validate_all_to_device(batch, device=device)
            m = model.forward(
                batch, epoch=total_epoch, total_iter=total_iter, is_training=True,
                logger=logger if log_image else None, logger_prefix="train"
            )
            model.backward()
            print(total_iter, end=' ')
            for name, loss in m.items():
                print(f"{name}: {loss.item()},", end=" ")
            print()
            if log_loss:
                if logger is not None:
                    for name, loss in m.items():
                        logger.add_scalar(f'train_loss/{name}', loss, total_iter)
            if log_image:
                model.set_eval()
                with torch.no_grad():
                    try:
                        batch = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_dataloader)
                        batch = next(test_iter)
                    batch = validate_all_to_device(batch, device=device)
                    m = model.forward(
                        batch, epoch=total_epoch, total_iter=total_iter, is_training=False, logger=logger,
                        logger_prefix="test")
                model.set_train()
                if logger is not None:
                    for name, loss in m.items():
                        logger.add_scalar(f'test_loss/{name}', loss, total_iter)
            if total_iter >= max_iter:
                break
    if logger is not None:
        logger.finish()
    return model


def validate_tensor_to_device(x, device=None):
    if type(x) is not torch.Tensor:
        return x
    elif torch.any(torch.isnan(x)):
        return None
    elif device is None:
        return x
    else:
        return x.to(device)


def validate_all_to_device(batch, device=None):
    return tuple(validate_tensor_to_device(x, device) for x in batch)


def render_bones(mvp, bones_pred, size=(256, 256), show_legend=False, overlay_img=None):
    bone_world4 = torch.concat([bones_pred, torch.ones_like(bones_pred[..., :1]).to(bones_pred.device)], dim=-1)
    b, f, num_bones = bone_world4.shape[:3]
    bones_clip4 = (bone_world4.view(b, f, num_bones*2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)).view(b, f, num_bones, 2, 4)
    bones_uv = bones_clip4[..., :2] / bones_clip4[..., 3:4]  # b, f, num_bones, 2, 2
    dpi = 32
    fx, fy = size[1] // dpi, size[0] // dpi
    rendered = []
    for b_idx in range(b):
        for f_idx in range(f):
            frame_bones_uv = bones_uv[b_idx, f_idx].cpu().numpy()
            fig = plt.figure(figsize=(fx, fy), dpi=dpi, frameon=False)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            if overlay_img is not None:
                bg_img = overlay_img[b_idx, f_idx].permute(1, 2, 0).cpu().numpy()  # (h, w, 3)
                bg_img = np.flipud(bg_img)
                ax.imshow(bg_img, extent=(-1, 1, -1, 1), alpha=0.5)  # Adjust extent to match plot coordinates
            colors = plt.cm.tab20(np.linspace(0, 1, frame_bones_uv.shape[0]))
            for bone_idx, bone in enumerate(frame_bones_uv):
                ax.plot(bone[:, 0], bone[:, 1], marker='o', linewidth=8, markersize=20, color=colors[bone_idx], label=str(bone_idx))
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.invert_yaxis()
            if show_legend:
                ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
            # Convert to image
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            w, h = fig.canvas.get_width_height()
            image.resize(h, w, 3)
            rendered += [image / 255.]
            plt.close(fig)

    rendered = expandBF(torch.FloatTensor(np.stack(rendered, 0)).permute(0, 3, 1, 2).to(bones_pred.device), b, f)
    return rendered


def project_points(points, mvp, normalize=True):  # Same as xfm_points without normalize
    """
    Project points from world space to clip space using mvp matrix
    points: (b, v, 4) or (b, v, 3)
    mvp: (b, 4, 4)
    """
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[..., :1], device=points.device)], dim=-1)
    points_clip4 = points @ mvp.transpose(-1, -2)
    if normalize:
        points_clip4 = points_clip4 / points_clip4[..., 3:4]
    return points_clip4


# def get_keypoint_coordinates(bones, mvp):
#     """
#     Input posed bones in world space and mvp matrix, output keypoint coordinates in clip space
#     head -> tail -> LF -> RF -> LR -> RR
#     Following Animal3D, normalize xy coordinate but keep z(depth) unormalized
#     """
#     keypoint_mapping = {  # keypoint index : (bone_index, bone_end_index(0 or 1))
#         0: [(0,1)], 1: [(0,0),(1,1)], 2: [(1,0),(2,1)], 3: [(2,0),(3,1),(10,0),(19,0)],  # head -> mid
#         4: [(3,0), (7,0)],  # mid
#         5: [(6,0),(7,1)], 6: [(5,0),(6,1),(13,0),(16,0)], 7: [(4,0),(5,1)], 8: [(4,1)],  # mid -> tail
#         9: [(9,0),(10,1)], 10: [(8,0),(9,1)], 11: [(8,1)],   # LF
#         12: [(18,0),(19,1)], 13: [(17,0),(18,1)], 14: [(17,1)],  # RF
#         15: [(12,0),(13,1)], 16: [(11,0),(12,1)], 17: [(11,1)],  # LR
#         18: [(15,0),(16,1)], 19: [(14,0),(15,1)], 20: [(14,1)],  # RR
#     }
#     bone_world4 = torch.concat([bones, torch.ones_like(bones[..., :1]).to(bones.device)], dim=-1)
#     b, f, num_bones = bone_world4.shape[:3]
#     bones_clip4 = (bone_world4.view(b, f, num_bones * 2, 1, 4) @ mvp.transpose(-1, -2).reshape(b, f, 1, 4, 4)).view(b,f, num_bones,2,4)
#     keypoint = torch.zeros((b, f, 21, 4), device=bones.device)
#     for k, v in keypoint_mapping.items():
#         for bone_idx in v:
#             keypoint[:, :, k, :] += bones_clip4[:, :, bone_idx[0], bone_idx[1], :]
#         keypoint[:, :, k, :] /= len(v)
#     # keypoint = torch.cat([keypoint[..., :2] / keypoint[..., 3:4], keypoint[..., 2:3]], dim=-1)  # Follow Animal3D
#     # keypoint = keypoint[..., :3] / keypoint[..., 3:4]  # Normalize all including depth
#     keypoint = torch.cat([keypoint[..., :3] / keypoint[..., 3:4], keypoint[..., 3:4]], dim=-1)  # Normalize xyz, keep original w
#     return keypoint


def get_keypoint_coordinates(bones, mvp):
    """
    Input posed bones in world space and mvp matrix, output keypoint coordinates in clip space
    head -> tail -> LF -> RF -> LR -> RR
    Following Animal3D, normalize xy coordinate but keep z(depth) unormalized
    """
    keypoint_mapping = {  # keypoint index : (bone_index, bone_end_index(0 or 1))
        0: [(0,1)], 1: [(0,0),(1,1)], 2: [(1,0),(2,1)], 3: [(2,0),(3,1),(10,0),(19,0)],  # head -> mid
        4: [(3,0), (7,0)],  # mid
        5: [(6,0),(7,1)], 6: [(5,0),(6,1),(13,0),(16,0)], 7: [(4,0),(5,1)], 8: [(4,1)],  # mid -> tail
        9: [(9,0),(10,1)], 10: [(8,0),(9,1)], 11: [(8,1)],   # LF
        12: [(18,0),(19,1)], 13: [(17,0),(18,1)], 14: [(17,1)],  # RF
        15: [(12,0),(13,1)], 16: [(11,0),(12,1)], 17: [(11,1)],  # LR
        18: [(15,0),(16,1)], 19: [(14,0),(15,1)], 20: [(14,1)],  # RR
    }
    b, f, k, v = bones.shape[:4]
    bones = rearrange(bones, "b f k v d -> (b f) (k v) d")
    assert bones.shape[0] == mvp.shape[0] == b * f
    bones_clip4 = xfm_points(bones, mvp)
    bones_clip4 /= bones_clip4[..., 3:4]
    bones_clip4 = rearrange(bones_clip4, "(b f) (k v) d -> b f k v d", b=b, f=f, k=k, v=v)
    keypoint = torch.zeros((b, f, 21, 4), device=bones.device)
    for k, v in keypoint_mapping.items():
        for bone_idx in v:
            keypoint[:, :, k, :] += bones_clip4[:, :, bone_idx[0], bone_idx[1], :]
        keypoint[:, :, k, :] /= len(v)
    return keypoint


def get_eval_aux(shape, mvp, resolution=(256, 256)):
    """
    Save all vertices in clip space project to screen, with visibility mask (..., 4)
    KT evaluation will map source gt keypoint to the nearest visible vertex in 2D and transfer to same vertex in target 2D
    PCK evaluation will later learn mapping 3D shape -> 3D keypoint -> 2D keypoint
    """
    visibility = torch.zeros((shape.v_pos.shape[0], shape.v_pos.shape[1]), device=shape.v_pos.device, dtype=torch.bool)
    v_pos_clip4 = xfm_points(shape.v_pos, mvp)
    v_pos_clip4 = v_pos_clip4 / v_pos_clip4[..., 3:]
    rast, _ = rasterize(RasterizeGLContext(), v_pos_clip4, shape.t_pos_idx[0].int(), resolution)
    face_ids = rast[..., -1]
    for i, (face_id, face) in enumerate(zip(face_ids, shape.t_pos_idx)):
        visible_faces = torch.unique(face_id)
        visible_faces = visible_faces[(visible_faces >= 0) & (visible_faces < face.shape[0])].long()
        visible_vertices = torch.unique(face[visible_faces])
        visibility[i][visible_vertices] = True
    eval_aux = torch.cat([v_pos_clip4[..., :3], visibility.unsqueeze(-1)], dim=-1)
    return eval_aux



class TensorDict(torch.nn.Module):
    """
    Custom tensor dictionary to store index-tensor mapping as key-value pairs
    """
    def __init__(self):
        super().__init__()
        self.tensor_dict = torch.nn.ParameterDict()
        self.previous_tensor_mean_dict = {}
        self.leg_bone_idx = [8,9,10,11,12,13,14,15,16,17,18,19]

    def __setitem__(self, keys, values):
        assert keys.shape[0] == values.shape[0]
        for key, value in zip(keys, values):
            key = str(int(key.item()))
            if key in self.tensor_dict:
                continue
            else:
                arti_params = value.clone().detach()
                # arti_params[self.leg_bone_idx] = torch.randn_like(arti_params[self.leg_bone_idx])
                self.tensor_dict[key] = torch.nn.Parameter(arti_params, requires_grad=True)
                self.previous_tensor_mean_dict[key] = value.mean().item()

    def __getitem__(self, keys):
        for key in keys:
            key = str(int(key.item()))
            prev_mean = self.previous_tensor_mean_dict[key]
            curr_mean = self.tensor_dict[key].mean().item()
            if prev_mean != curr_mean:
                # print(f"TensorDict: {key} has changed, diff={curr_mean-prev_mean}", flush=True)
                self.previous_tensor_mean_dict[key] = self.tensor_dict[key].mean().item()
            else:
                # print(f"TensorDict: {key} has not changed", flush=True)
                pass
        return torch.stack([self.tensor_dict[str(int(key.item()))] for key in keys], dim=0)

    def items(self):
        return self.tensor_dict.items()

    def forward(self, indices):
        return self.__getitem__(indices)