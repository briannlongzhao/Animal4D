import sys
import os
import numpy as np
import cv2
import pickle as pkl
import torch
import imageio
import trimesh
from glob import glob
from tqdm import trange
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.renderer import Textures
from scipy.spatial.transform import Rotation as R
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor
from models.utils import get_all_sequence_dirs, images_to_video
submodule_path = "externals/SMALify"
sys.path.insert(0, submodule_path)
sys.path.insert(0, os.path.join(submodule_path, "smal_fitter"))
from externals.SMALify import config
config.BADJA_PATH = os.path.join(submodule_path, config.BADJA_PATH)
config.SMAL_DATA_FILE = os.path.join(submodule_path, config.SMAL_DATA_FILE)
config.UNITY_SHAPE_PRIOR = os.path.join(submodule_path, config.UNITY_SHAPE_PRIOR)
config.WALKING_PRIOR_FILE = os.path.join(submodule_path, config.WALKING_PRIOR_FILE)
config.SMAL_FILE = os.path.join(submodule_path, config.SMAL_FILE)
config.SMAL_MODEL_PATH = os.path.join(submodule_path, config.SMAL_MODEL_PATH)
config.SMAL_SYM_FILE = os.path.join(submodule_path, config.SMAL_SYM_FILE)
config.SMAL_UV_FILE = os.path.join(submodule_path, config.SMAL_UV_FILE)
config.IMAGE_RANGE = None
config.OUTPUT_DIR = os.path.join(submodule_path, config.OUTPUT_DIR)
config.SHAPE_FAMILY = 1
config.TORSO_JOINTS = [3, 4, 5, 8, 11, 14]
# mapping from data keypoint idx to SMAL joint idx
config.CANONICAL_MODEL_JOINTS = [39, 40, 35, 15, 25, 8, 9, 10, 12, 13, 14, 18, 19, 20, 22, 23, 24]
config.MESH_COLOR = [255, 255, 255]
config.MARKER_COLORS = [
    [230, 25, 75], [230, 25, 75], [230, 25, 75],  # face red
    [255, 255, 25], [255, 255, 25],  # body yellow
    [60, 180, 75], [60, 180, 75], [60, 180, 75],  # LF green
    [245, 130, 48], [245, 130, 48], [245, 130, 48],  # RF blue
    [240, 50, 230], [240, 50, 230], [240, 50, 230],  # LR magenta
    [255, 153, 204], [255, 153, 204], [255, 153, 204],  # RR pink
]
config.SKELETON = [
    [0, 1], [1, 2], [0, 2],
    [2, 3], [3, 4],
    [3, 5], [5, 6], [6, 7],
    [3, 8], [8, 9], [9, 10],
    [4, 11], [11, 12], [12, 13],
    [4, 14], [14, 15], [15, 16]
]
sys.modules["config"] = config
from externals.SMALify.smal_fitter.smal_fitter import SMALFitter
from externals.SMALify.smal_fitter.data_loader import load_badja_sequence, load_stanford_sequence
from externals.SMALify.smal_fitter.p3d_renderer import Renderer



# data_dir = "data/data_3.0.0/train/racoon/7Hbt9Oncz1M_003_002"
data_dir = "data/data_3.0.0"


class ImageExporter:
    def __init__(self, output_dir, filenames):
        self.output_dir = output_dir
        self.file_names = filenames
        self.stage_id = 0
        self.epoch_name = 0
        self.image_suffix = "rgb.png"
        self.mesh_suffix = "mesh_smalify.ply"
        self.rgb_overlayed_suffix = "rgb_overlayed_smalify.png"
        self.mask_diff_suffix = "mask_diff_smalify.png"
        self.mask_pred_suffix = "mask_smalify.png"
        self.shading_suffix = "shading_smalify.png"
        self.shading_bones_suffix = "shading_bones_smalify.png"
        self.kt_aux_suffix = "kt_aux_smalify.txt"
        self.keypoint_suffix = "keypoint_smalify.txt"

    def export(
        self, global_id, vertices, faces, rgb_overlayed, mask_pred, mask_diff, shading, shading_bones, kt_aux,
        keypoint_pred
    ):
        filename = os.path.join(self.output_dir, self.file_names[global_id])
        imageio.imsave(filename.replace(self.image_suffix, self.rgb_overlayed_suffix), rgb_overlayed)
        imageio.imsave(filename.replace(self.image_suffix, self.mask_diff_suffix), mask_diff)
        imageio.imsave(filename.replace(self.image_suffix, self.mask_pred_suffix), mask_pred)
        imageio.imsave(filename.replace(self.image_suffix, self.shading_suffix), shading)
        imageio.imsave(filename.replace(self.image_suffix, self.shading_bones_suffix), shading_bones)
        mesh = trimesh.Trimesh(vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy(), process=False)
        mesh.export(filename.replace(self.image_suffix, self.mesh_suffix))
        with open(filename.replace(self.image_suffix, self.kt_aux_suffix), "w") as f:
            np.savetxt(f, kt_aux.cpu().numpy(), fmt="%.6f", delimiter=' ')
        with open(filename.replace(self.image_suffix, self.keypoint_suffix), "w") as f:
            np.savetxt(f, keypoint_pred.cpu().numpy(), fmt="%.6f", delimiter=' ')


class CustomRenderer(Renderer):
    def __init__(self, image_size, device):
        super(CustomRenderer, self).__init__(image_size, device)

    def forward(self, vertices, points, faces, render_texture=False):
        tex = torch.ones_like(vertices) * self.mesh_color # (1, V, 3)
        textures = Textures(verts_rgb=tex)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        sil_images = self.silhouette_renderer(mesh)[..., -1].unsqueeze(1)
        screen_size = torch.ones(vertices.shape[0], 2).to(vertices.device) * self.image_size
        screen_size = screen_size[0].unsqueeze(0)
        proj_points = self.cameras.transform_points_screen(points, image_size=screen_size)[:, :, [1, 0]]

        if render_texture:
            color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
            return sil_images, proj_points, color_image
        else:
            return sil_images, proj_points

    def render_with_kt_aux(self, vertices, points, faces):
        tex = torch.ones_like(vertices) * self.mesh_color  # (1, V, 3)
        textures = Textures(verts_rgb=tex)
        mesh = Meshes(verts=vertices, faces=faces, textures=textures)
        sil_images = self.silhouette_renderer(mesh)[..., -1].unsqueeze(1)
        screen_size = torch.ones(vertices.shape[0], 2).to(vertices.device) * self.image_size
        screen_size = screen_size[0].unsqueeze(0)
        proj_points = self.cameras.transform_points_screen(points, image_size=screen_size)
        color_image = self.color_renderer(mesh).permute(0, 3, 1, 2)[:, :3, :, :]
        # Visibility
        rasterizer = self.color_renderer.rasterizer
        fragments = rasterizer(mesh)
        face_ids = fragments.pix_to_face[..., 0]
        visibility = torch.zeros((vertices.shape[0], vertices.shape[1]), dtype=torch.bool, device=vertices.device)
        for i, (face_id, face) in enumerate(zip(face_ids, faces)):
            visible_faces = torch.unique(face_id)
            visible_faces = visible_faces[(visible_faces >= 0) & (visible_faces < face.shape[0])]
            visible_vertices = torch.unique(face[visible_faces])
            visibility[i][visible_vertices] = True
        proj_verts = self.cameras.transform_points_screen(vertices, image_size=screen_size)[:, :, [0, 1]]
        proj_verts = proj_verts / screen_size * 2 - 1
        kt_aux = torch.cat([proj_verts, visibility[..., None]], dim=-1)
        return sil_images, proj_points, color_image, kt_aux




class CustomSMALFitter(SMALFitter):
    def __init__(self, device, data, window_size, shape_family, use_unity_prior):
        super(CustomSMALFitter, self).__init__(device, data, window_size, shape_family, use_unity_prior)
        self.renderer = CustomRenderer(self.image_size, device)

    def generate_visualization(self, image_exporter):
        rot_matrix = torch.from_numpy(R.from_euler('y', 180.0, degrees=True).as_matrix()).float().to(self.device)
        for j in range(0, self.num_images, self.batch_size):
            batch_range = list(range(j, min(self.num_images, j + self.batch_size)))
            batch_params = {
                'global_rotation': self.global_rotation[batch_range] * self.global_mask,
                'joint_rotations': self.joint_rotations[batch_range] * self.rotation_mask,
                'betas': self.betas.expand(len(batch_range), self.n_betas),
                'log_betascale': self.log_beta_scales.expand(len(batch_range), 6),
                'trans': self.trans[batch_range],
            }
            target_visibility = self.target_visibility[batch_range]
            rgb_imgs = self.rgb_imgs[batch_range].to(self.device)
            sil_imgs = self.sil_imgs[batch_range].to(self.device)
            with torch.no_grad():
                verts, joints, Rs, v_shaped = self.smal_model(
                    batch_params['betas'],
                    torch.cat([
                        batch_params['global_rotation'].unsqueeze(1),
                        batch_params['joint_rotations']], dim=1),
                    betas_logscale=batch_params['log_betascale'])
                verts = verts + batch_params['trans'].unsqueeze(1)
                joints = joints + batch_params['trans'].unsqueeze(1)
                canonical_joints = joints[:, config.CANONICAL_MODEL_JOINTS]
                rendered_silhouettes, rendered_joints, rendered_images, kt_aux = self.renderer.render_with_kt_aux(
                    verts, canonical_joints, self.smal_model.faces.unsqueeze(0).expand(verts.shape[0], -1, -1),
                )
                mask_pred = (rendered_silhouettes.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                mask_pred = np.repeat(mask_pred, 3, axis=-1)
                mask_diff = F.l1_loss(sil_imgs, rendered_silhouettes, reduction='none')
                mask_diff = (mask_diff.expand_as(rgb_imgs).data.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                overlay_image = (rendered_images * rendered_silhouettes) + (rgb_imgs * (1 - rendered_silhouettes))
                overlay_image = draw_joints(overlay_image, rendered_joints)
                overlay_image = (overlay_image.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                rendered_images_bones = draw_joints(rendered_images, rendered_joints)
                rendered_images_bones = (rendered_images_bones.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                rendered_images = (rendered_images.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                rendered_joints[:, :, :2] = (rendered_joints[:, :, :2] / torch.tensor(rendered_images.shape[1:-1], device=rendered_joints.device)) * 2 - 1
                for batch_id, global_id in enumerate(batch_range):
                    image_exporter.export(
                        global_id=global_id,
                        vertices=verts[batch_id], faces=self.smal_model.faces.data,
                        rgb_overlayed=overlay_image[batch_id],
                        mask_diff=mask_diff[batch_id],
                        mask_pred=mask_pred[batch_id],
                        shading=rendered_images[batch_id],
                        shading_bones=rendered_images_bones[batch_id],
                        kt_aux=kt_aux[batch_id],
                        keypoint_pred=rendered_joints[batch_id],
                    )



def draw_joints(image, landmarks, visible=None):
    def draw_joints_np(image_np, landmarks_np, visible_np=None):
        image_np = (image_np * 255.0).astype(np.uint8)
        bs, nj, _ = landmarks_np.shape
        if visible_np is None:
            visible_np = np.ones((bs, nj), dtype=bool)
        return_images = []
        for image_sgl, landmarks_sgl, visible_sgl in zip(image_np, landmarks_np, visible_np):
            image_sgl = image_sgl.copy()
            inv_ctr = 0
            for joint_a, joint_b in config.SKELETON:
                if visible_sgl[joint_a] and visible_sgl[joint_b]:
                    y_a, x_a = map(int, landmarks_sgl[joint_a])
                    y_b, x_b = map(int, landmarks_sgl[joint_b])
                    cv2.line(image_sgl, (x_a, y_a), (x_b, y_b), (255, 255, 0), 2)
            for joint_id, ((y_co, x_co), vis) in enumerate(zip(landmarks_sgl, visible_sgl)):
                color = np.array(config.MARKER_COLORS)[joint_id]
                marker_type = np.array(config.MARKER_TYPE)[joint_id]
                if not vis:
                    x_co, y_co = inv_ctr * 10, 0
                    inv_ctr += 1
                cv2.drawMarker(
                    image_sgl, (int(x_co), int(y_co)), (int(color[0]), int(color[1]), int(color[2])),
                    marker_type, 8, thickness=3
                )
            return_images.append(image_sgl)
        return_stack = np.stack(return_images, 0)
        return_stack = return_stack / 255.0
        return return_stack

    image_np = np.transpose(image.cpu().data.numpy(), (0, 2, 3, 1))
    landmarks_np = landmarks[:, :, [1, 0]].cpu().data.numpy()
    if visible is not None:
        visible_np = visible.cpu().data.numpy()
    else:
        visible_np = visible
    return_stack = draw_joints_np(image_np, landmarks_np, visible_np)
    return torch.FloatTensor(np.transpose(return_stack, (0, 3, 1, 2)))


def load_custom_sequence(
    sequence_dir, in_size=512, resize=256, image_range=None,
    rgb_suffix="rgb.png", mask_suffix="mask.png", keypoint_suffix="keypoint.txt"
):
    all_rgbs = sorted(glob(os.path.join(sequence_dir, f"*{rgb_suffix}")))
    all_filenames = [os.path.basename(path) for path in all_rgbs]
    all_paths = [path.replace(rgb_suffix, "{}") for path in all_rgbs]
    all_masks = [path.format(mask_suffix) for path in all_paths]
    all_keypoints = [path.format(keypoint_suffix) for path in all_paths]
    transform = Compose([Resize((resize, resize), interpolation=InterpolationMode.BILINEAR), ToTensor()])
    all_rgbs = torch.stack([transform(default_loader(path)) for path in all_rgbs])
    all_masks = torch.stack([transform(default_loader(path))[[0]] for path in all_masks])
    all_keypoints_raw = torch.stack([torch.from_numpy(np.loadtxt(path)).float() for path in all_keypoints])
    all_keypoints = all_keypoints_raw[:, :, [1,0]] / in_size * resize
    all_visibility = (all_keypoints_raw[:, :, 2] > 0.5).int()
    if image_range is not None:
        all_rgbs = all_rgbs[image_range]
        all_masks = all_masks[image_range]
        all_keypoints = all_keypoints[image_range]
        all_visibility = all_visibility[image_range]
        all_filenames = [all_filenames[i] for i in image_range]
    print(f"Loaded {len(all_filenames)} images from {sequence_dir}")
    return (all_rgbs, all_masks, all_keypoints, all_visibility), all_filenames


def main(sequence_dir):
    if len(list(glob(os.path.join(sequence_dir, "*smalify.png")))) > 0:
        return
    category = os.path.dirname(sequence_dir).split("/")[-1]
    if category in ["cat", "cougar", "tiger", "leopard", "panther"]:
        config.SHAPE_FAMILY = 0
    elif category in ["dog", "bear", "boar", "fox", "pig", "rabbit", "racoon", "wolf"]:
        config.SHAPE_FAMILY = 1
    elif category in ["deer", "horse", "moose", "zebra"]:
        config.SHAPE_FAMILY = 2
    elif category in ["cow", "elephant", "goat", "sheep"]:
        config.SHAPE_FAMILY = 3
    elif category in ["hippo", "rhino"]:
        config.SHAPE_FAMILY = 4
    else:
        raise NotImplementedError
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.GPU_IDS
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset, name = config.SEQUENCE_OR_IMAGE_NAME.split(":")
    dataset = "custom"
    if dataset == "badja":
        data, filenames = load_badja_sequence(
            config.BADJA_PATH, name,
            config.CROP_SIZE, image_range=config.IMAGE_RANGE)
    elif dataset == "stanford_extra":
        data, filenames = load_stanford_sequence(
            config.STANFORD_EXTRA_PATH, name,
            config.CROP_SIZE
        )
    elif dataset == "custom":
        data, filenames = load_custom_sequence(sequence_dir=sequence_dir, image_range=config.IMAGE_RANGE)
    else:
        raise NotImplementedError
    dataset_size = len(filenames)
    print("Dataset size: {0}".format(dataset_size))
    assert config.SHAPE_FAMILY >= 0, "Shape family should be greater than 0"
    use_unity_prior = config.SHAPE_FAMILY == 1 and not config.FORCE_SMAL_PRIOR
    use_unity_prior = True
    if not use_unity_prior and not config.ALLOW_LIMB_SCALING:
        print(
            "WARNING: Limb scaling is only recommended for the new Unity prior. TODO: add a regularizer to constrain scale parameters.")
        config.ALLOW_LIMB_SCALING = False
    image_exporter = ImageExporter(sequence_dir, filenames)
    model = CustomSMALFitter(device, data, config.WINDOW_SIZE, config.SHAPE_FAMILY, use_unity_prior)
    for stage_id, weights in enumerate(np.array(config.OPT_WEIGHTS).T):
        opt_weight = weights[:6]
        w_temp = weights[6]
        epochs = int(weights[7])
        lr = weights[8]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
        if stage_id == 0:
            model.joint_rotations.requires_grad = False
            model.betas.requires_grad = False
            model.log_beta_scales.requires_grad = False
            target_visibility = model.target_visibility.clone()
            model.target_visibility *= 0
            model.target_visibility[:, config.TORSO_JOINTS] = target_visibility[:, config.TORSO_JOINTS]  # Turn on only torso points
        else:
            model.joint_rotations.requires_grad = True
            model.betas.requires_grad = True
            if config.ALLOW_LIMB_SCALING:
                model.log_beta_scales.requires_grad = True
            model.target_visibility = data[-1].clone()
        t = trange(epochs, leave=True)
        for epoch_id in t:
            image_exporter.stage_id = stage_id
            image_exporter.epoch_name = str(epoch_id)
            acc_loss = 0
            optimizer.zero_grad()
            for j in range(0, dataset_size, config.WINDOW_SIZE):
                batch_range = list(range(j, min(dataset_size, j + config.WINDOW_SIZE)))
                loss, losses = model(batch_range, opt_weight, stage_id)
                acc_loss += loss.mean()
                # print ("Optimizing Stage: {}\t Epoch: {}, Range: {}, Loss: {}, Detail: {}".format(stage_id, epoch_id, batch_range, loss.data, losses))

            joint_loss, global_loss, trans_loss = model.get_temporal(w_temp)
            desc = "EPOCH: Optimizing Stage: {}\t Epoch: {}, Loss: {:.2f}, Temporal: ({}, {}, {})".format(
                stage_id, epoch_id,
                acc_loss.data, joint_loss.data,
                global_loss.data, trans_loss.data)
            t.set_description(desc)
            t.refresh()
            acc_loss = acc_loss + joint_loss + global_loss + trans_loss
            acc_loss.backward()
            optimizer.step()

        #     break
        # break

    image_exporter.stage_id = 10
    image_exporter.epoch_name = str(0)
    model.generate_visualization(image_exporter)

    # Create video # TODO: need ffmpeg with h264 support
    # mask_diff_files = sorted(glob(os.path.join(sequence_dir, "*mask_diff_smalify.png")))
    # images_to_video(mask_diff_files, os.path.join(sequence_dir, "mask_diff_smalify.mp4"))
    # rgb_overlayed_files = sorted(glob(os.path.join(sequence_dir, "*rgb_overlayed_smalify.png")))
    # images_to_video(rgb_overlayed_files, os.path.join(sequence_dir, "rgb_overlayed_smalify.mp4"))

    # shading_files = sorted(glob(os.path.join(sequence_dir, "*shading_smalify.png")))
    # images_to_video(shading_files, os.path.join(sequence_dir, "shading_smalify.mp4"))
    # shading_bones_files = sorted(glob(os.path.join(sequence_dir, "*shading_bones_smalify.png")))
    # images_to_video(shading_bones_files, os.path.join(sequence_dir, "shading_bones_smalify.mp4"))




if __name__ == '__main__':
    all_sequence_dirs = get_all_sequence_dirs(data_dir)
    for sequence_dir in all_sequence_dirs:
        print("Processing", sequence_dir)
        main(sequence_dir)
