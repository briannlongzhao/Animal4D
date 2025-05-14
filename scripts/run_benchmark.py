import os
import shutil

import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from time import time
from einops import repeat
from collections import Counter
from configargparse import ArgumentParser
from torch.nn import Parameter, Module, MSELoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate


from models.utils import get_all_sequence_dirs, draw_keypoints

all_methods = [
    "fauna++",
    # "fauna",
    # "smalify"
]


local_dir = "/scr-ssd/briannlz"
if not os.path.exists(local_dir):
    local_dir = "/scr/briannlz"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to benchmark data directory")
    # parser.add_argument("--image_size", type=int, default=512, help="Input image size for evaluation")
    return parser.parse_args()



def mask_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum()
    union = torch.logical_or(mask1, mask2).sum()
    return intersection / union


def mpjpe(pred, gt, normalize=False, image_size=512):
    """Input should be in image pixel space"""
    if normalize:
        pred = pred / image_size
        gt = gt / image_size
    return torch.mean(torch.norm(pred - gt, dim=-1))


def mpjve(pred, gt):
    """Input should be in [-1,1] and have at least 3 dimensions (t, k, d)"""
    pred_velocity = pred[..., 1:, :, :] - pred[..., :-1, :, :]
    gt_velocity = gt[..., 1:, :, :] - gt[..., :-1, :, :]
    return torch.mean(torch.norm(pred_velocity - gt_velocity, dim=-1))


def pck(pred, gt, norm_dist, threshold=0.1):
    """Normalized by sqrt of gt mask area"""
    dist_threshold = threshold * norm_dist
    dist_threshold = repeat(dist_threshold, "t -> t k", k=gt.shape[-2])
    dists = torch.norm(pred - gt, dim=-1)
    return torch.mean((dists < dist_threshold).float())


def kt(src_verts, src_visibility, src_kp, tgt_verts, tgt_kp, image_size, norm_dist, threshold=0.1, src_img=None, tgt_img=None):
    src_verts[src_visibility.bool() == 0] = torch.inf
    dists = torch.norm(src_verts[None, :,:] - src_kp[:, None, :], dim=-1)
    vert_ids = torch.argmin(dists, dim=1)
    pred_kp = tgt_verts[vert_ids]
    if src_img is not None and tgt_img is not None:
        src_kp_img = draw_keypoints(image=src_img, keypoint=(src_kp+1)/2)
        tgt_kp_img = draw_keypoints(image=tgt_img, keypoint=(pred_kp+1)/2)
    pred_kp = (pred_kp.unsqueeze(0) + 1) * 0.5 * image_size[0]
    tgt_kp = (tgt_kp.unsqueeze(0) + 1) * 0.5 * image_size[0]
    return pck(pred_kp.unsqueeze(0), tgt_kp.unsqueeze(0), norm_dist=norm_dist, threshold=threshold)


class Mapping(Module):
    def __init__(self, n_vtx, n_kp=17, init_type='uniform'):
        super(Mapping, self).__init__()
        self.vertix_num = n_vtx
        self.init_type = init_type
        if init_type == 'randn':
            self.M = Parameter(torch.randn(1, n_kp, self.vertix_num).to(torch.float32))
        elif init_type == 'uniform':
            self.M = Parameter(torch.ones(1, n_kp, self.vertix_num).to(torch.float32)/self.vertix_num)

        self.vis_posed_verts = None
        self.vis_pred_ldmks = None
        self.vis_imgs = None
        self.vis_gt_ldmks = None
        self.vis_recon_imgs = None
        self.vis_mvps = None
        self.vis_homogeneous_x = None

    def forward(self, x, mvps=None, do_visualize=False):
        if do_visualize:
            self.posed_verts = self.project(x, mvps)
        M = torch.max(self.M, torch.zeros_like(self.M))
        M = M/M.sum(dim=-1, keepdim=True)

        x = torch.bmm(M.expand(len(x),-1,-1), x)
        # self.pred_ldmks = self.project(x, mvps)
        return x

    def homogenize(self, x):
        return torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)

    def project(self, x, mvps):
        verts_homogeneous = self.homogenize(x)
        bs, v_num, v_dim = verts_homogeneous.shape
        posed_verts_homogeneous = (verts_homogeneous.view(bs, v_num, 1, v_dim) @ mvps.transpose(-1, -2).reshape(bs, 1, 4, 4)).view(bs, v_num, 4)
        posed_verts = posed_verts_homogeneous[...,:2]/posed_verts_homogeneous[...,3:4]
        return posed_verts




class VertexMappingDataset(Dataset):
    def __init__(self, category_dir, method, image_size, n_vtx, n_kp=17, local_dir=None):
        self.category_dir = category_dir
        self.all_kp_gt = sorted(glob(f"{category_dir}/**/*keypoint.txt", recursive=True))
        self.all_eval_aux = sorted(glob(f"{category_dir}/**/*eval_aux_{method}.txt", recursive=True))
        self.n_vtx = n_vtx
        self.image_size = image_size
        if local_dir is not None:
            print("Copying data to local directory")
            os.makedirs(local_dir, exist_ok=True)
            for i in range(len(self.all_kp_gt)):
                src_kp_gt = self.all_kp_gt[i]
                src_eval_aux = self.all_eval_aux[i]
                dst_kp_gt = os.path.join(local_dir, src_kp_gt.lstrip("/"))
                dst_eval_aux = os.path.join(local_dir, src_eval_aux.lstrip("/"))
                os.makedirs(os.path.dirname(dst_kp_gt), exist_ok=True)
                os.makedirs(os.path.dirname(dst_eval_aux), exist_ok=True)
                shutil.copy(src_kp_gt, dst_kp_gt)
                shutil.copy(src_eval_aux, dst_eval_aux)
                self.all_kp_gt[i] = dst_kp_gt
                self.all_eval_aux[i] = dst_eval_aux
        assert len(self.all_kp_gt) == len(self.all_eval_aux)


    def __len__(self):
        return len(self.all_kp_gt)

    def __getitem__(self, idx):
        kp_gt = torch.from_numpy(np.loadtxt(self.all_kp_gt[idx], delimiter=" ").astype(np.float32))[:, :2]
        kp_gt = kp_gt / self.image_size * 2 - 1
        verts = torch.from_numpy(np.loadtxt(self.all_eval_aux[idx], delimiter=" ").astype(np.float32))[:, :3]
        if len(verts) < self.n_vtx:
            verts = torch.cat([verts, torch.zeros(self.n_vtx - len(verts), 3)])
        if len(verts) > self.n_vtx:
            verts = verts[:self.n_vtx]
        return verts, kp_gt


def train_vtx_to_kp_mapping(category_dir, mapping_path, method, device, n_vtx, n_kp=17):
    mapping_dataset = VertexMappingDataset(
        category_dir=category_dir, method=method, image_size=target_size[0], n_vtx=n_vtx, n_kp=17, local_dir=local_dir
    )
    mapping_dataloader = DataLoader(
        mapping_dataset, batch_size=8, num_workers=8, shuffle=True, pin_memory=True
    )
    model = Mapping(n_vtx, n_kp).to(device)
    optimizer = Adam(model.parameters(), lr=0.00001)
    model.train()
    loss_fn = MSELoss()
    for epoch in tqdm(range(1000)):
        total_loss = 0
        for batch in mapping_dataloader:
            verts = batch[0].to(device)
            kp_gt = batch[1].to(device)
            optimizer.zero_grad()
            output = model(verts)
            loss = loss_fn(output[..., :2], kp_gt)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(mapping_dataloader)
        print(f"Epoch {epoch}: Loss {total_loss}")
        os.makedirs("data/temp_visualization", exist_ok=True)
    torch.save(model.state_dict(), mapping_path)
    model.eval()
    return model


def load_vtx_to_kp_mapping(mapping_path, device, n_vtx, n_kp=17):
    model = Mapping(n_vtx, n_kp).to(device)
    model.load_state_dict(torch.load(mapping_path))
    model.eval()
    return model


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_sequence_dirs = get_all_sequence_dirs(args.data_dir)
    # all_sequence_dirs = sorted(all_sequence_dirs)
    for method in all_methods:
        num_frames = []
        num_tracks = len(all_sequence_dirs)
        all_iou = []
        all_pck_01 = []
        all_pck_005 = []
        all_kt_01 = []
        all_kt_005 = []
        all_mpjve = []
        current_category = None
        mapping = None
        for sequence_dir in tqdm(all_sequence_dirs, desc=f"evaluating {method}"):
            print("Evaluating", sequence_dir)
            # maks iou
            start = time()
            mask_pred = sorted(glob(f"{sequence_dir}/*mask_{method}.png", recursive=True))
            mask_gt = sorted(glob(f"{sequence_dir}/*mask.png", recursive=True))
            assert len(mask_pred) == len(mask_gt), sequence_dir
            num_frames.append(len(mask_pred))
            mask_gt = torch.stack([
                torch.tensor(cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127, dtype=torch.uint8) for f in mask_gt
            ])
            target_size = (mask_gt.shape[-1], mask_gt.shape[-2])
            mask_pred = torch.stack([
                torch.tensor(cv2.resize(
                    (cv2.imread(f, cv2.IMREAD_GRAYSCALE) > 127).astype("uint8"), target_size, interpolation=cv2.INTER_NEAREST
                ), dtype=torch.uint8) for f in mask_pred
            ])
            mask_pred = mask_pred.to(device)
            mask_gt = mask_gt.to(device)
            all_iou.append(mask_iou(mask_pred, mask_gt).item())
            pck_norm_dist = torch.sqrt(torch.sum(mask_gt, dim=(-1, -2)))
            del mask_pred, mask_gt
            # print(f"Mask IoU took {(time() - start)/num_frames[-1]:.2f} seconds per frame")

            # pck
            start = time()
            eval_aux = sorted(glob(f"{sequence_dir}/*eval_aux_{method}.txt", recursive=True))
            kp_gt = sorted(glob(f"{sequence_dir}/*keypoint.txt", recursive=True))
            assert len(eval_aux) == len(kp_gt) == num_frames[-1], sequence_dir
            category_dir = os.path.dirname(sequence_dir)
            mapping_path = os.path.join(category_dir, f"mapping_{method}.pth")
            category_eval_aux = sorted(
                glob(os.path.join(category_dir, "**", f"*eval_aux_{method}.txt"), recursive=True))
            counts = [len(np.loadtxt(f, dtype=str, delimiter=' ')) for f in category_eval_aux]
            n_vtx = Counter(counts).most_common(1)[0][0]
            if not os.path.exists(mapping_path):
                mapping = train_vtx_to_kp_mapping(
                    category_dir=category_dir, mapping_path=mapping_path, method=method, device=device, n_vtx=n_vtx
                )
            else:
                mapping = load_vtx_to_kp_mapping(mapping_path, device, n_vtx=n_vtx)
                # if os.path.basename(category_dir) != current_category:
                #     mapping = load_vtx_to_kp_mapping(mapping_path, device, n_vtx=n_vtx)
            current_category = os.path.basename(category_dir)

            # kp_pred = sorted(glob(f"{sequence_dir}/*keypoint_{method}.txt", recursive=True))
            kp_gt = torch.stack([
                torch.from_numpy(np.loadtxt(f, delimiter=" ")).float().to(device) for f in kp_gt
            ])[..., :2]
            kp_pred = []
            for f in eval_aux:
                verts = torch.from_numpy(np.loadtxt(f, delimiter=" ")[:,:3]).float().to(device)
                if verts.shape[0] < n_vtx:
                    verts = torch.cat([verts, torch.zeros(n_vtx - verts.shape[0], 3).to(device)])
                if verts.shape[0] > n_vtx:
                    verts = verts[:n_vtx]
                kp_pred.append(mapping(verts.unsqueeze(0))[..., :2])
            kp_pred = torch.concat(kp_pred, dim=0)
            kp_pred = (kp_pred + 1) * 0.5 * target_size[0]
            # Overlapping joints: nose, withers, tail start, all leg bottom 2 joints LF, RF, LR, RR
            # kp_gt_overlap = kp_gt[..., [2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16], :]
            # if "fauna" in method:
            #     kp_pred_overlap = kp_pred[..., [0, 3, 7, 10, 11, 13, 14, 16, 17, 19, 20], :]
            # elif method == "smalify":
            #     kp_pred_overlap = kp_pred[..., [2, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16], :]
            # else:
            #     raise NotImplementedError
            all_pck_01.append(pck(kp_pred, kp_gt, pck_norm_dist, threshold=0.1).item())
            all_pck_005.append(pck(kp_pred, kp_gt, pck_norm_dist, threshold=0.05).item())
            # print(f"PCK took {(time() - start)/num_frames[-1]:.2f} seconds per frame")

            # keypoint transfer
            start = time()
            src_idx = 0
            tgt_idx = -1
            src_verts = torch.from_numpy(np.loadtxt(eval_aux[src_idx], delimiter=" ").astype(np.float32))[:, :2].to(device)
            tgt_verts = torch.from_numpy(np.loadtxt(eval_aux[tgt_idx], delimiter=" ").astype(np.float32))[:, :2].to(device)
            src_visibility = torch.from_numpy(np.loadtxt(eval_aux[src_idx], delimiter=" ").astype(np.float32))[:, -1].to(device)
            # tgt_visibility = torch.from_numpy(np.loadtxt(kp_aux[tgt_idx], delimiter=" ").astype(np.float32))[:, -1]
            src_kp = kp_gt[src_idx] / target_size[0] * 2 - 1
            tgt_kp = kp_gt[tgt_idx] / target_size[0] * 2 - 1
            all_images = sorted(glob(f"{sequence_dir}/*rgb.png", recursive=True))
            # For visualization
            src_img = torch.from_numpy(
                cv2.cvtColor(cv2.imread(all_images[src_idx]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            ).permute(2, 0, 1)
            tgt_img = torch.from_numpy(
                cv2.cvtColor(cv2.imread(all_images[tgt_idx]), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            ).permute(2, 0, 1)
            all_kt_01.append(kt(
                src_verts, src_visibility, src_kp, tgt_verts, tgt_kp,
                norm_dist=pck_norm_dist[None, tgt_idx], threshold=0.1, image_size=target_size,
                src_img=src_img, tgt_img=tgt_img
            ).item())
            all_kt_005.append(kt(
                src_verts, src_visibility, src_kp, tgt_verts, tgt_kp,
                norm_dist=pck_norm_dist[None, tgt_idx], threshold=0.05, image_size=target_size
            ).item())
            # print(f"KT took {(time() - start)/num_frames[-1]:.2f} seconds per frame")

            # MPJVE
            start = time()
            kp_pred = (kp_pred / target_size[0] * 2) - 1
            kp_gt = (kp_gt / target_size[0] * 2) - 1
            all_mpjve.append(mpjve(kp_pred, kp_gt).item())
            # print(f"MPJVE took {(time() - start)/num_frames[-1]:.2f} seconds per frame")


        weighted_iou = np.sum(np.array(num_frames) * np.array(all_iou)) / np.sum(num_frames)
        weighted_pck_01 = np.sum(np.array(num_frames) * np.array(all_pck_01)) / np.sum(num_frames)
        weighted_pck_005 = np.sum(np.array(num_frames) * np.array(all_pck_005)) / np.sum(num_frames)
        weighted_kt_01 = np.sum(np.array(num_frames) * np.array(all_kt_01)) / np.sum(num_frames)
        weighted_kt_005 = np.sum(np.array(num_frames) * np.array(all_kt_005)) / np.sum(num_frames)
        weighted_mpjve = np.sum(np.array(num_frames) * np.array(all_mpjve)) / np.sum(num_frames)

        print(f"{method}:")
        print("IoU: ", weighted_iou)
        print("PCK@0.1: ", weighted_pck_01)
        print("PCK@0.05: ", weighted_pck_005)
        print("KT@0.1: ", weighted_kt_01)
        print("KT@0.05: ", weighted_kt_005)
        print("MPJVE: ", weighted_mpjve)



