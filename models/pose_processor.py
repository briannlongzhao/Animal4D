import os.path
import time
import cv2
from glob import glob
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset, DataLoader

from models.pose_models import ViTPose



class PoseProcessor:
    def __init__(
        self, pose_method, image_suffix, keypoint_suffix, pose_image_suffix, batch_size, device, save_pose_image=True
    ):
        self.image_suffix = image_suffix
        self.keypoint_suffix = keypoint_suffix
        self.pose_image_suffix = pose_image_suffix
        self.batch_size = batch_size
        assert self.batch_size == 1, "only support 1 numpy image input"
        self.device = device
        self.save_pose_image = save_pose_image
        if pose_method == "vitpose":
            cfg_path = "externals/ViTPose/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py"
            ckpt_path = "externals/ViTPose/ckpt/apt36k.pth"
            self.model = ViTPose(
                cfg_path=cfg_path, ckpt_path=ckpt_path, device=device, return_pose_image=save_pose_image
            )
        else:
            raise NotImplementedError

    def run_track(self, track_dir):
        dataloader = DataLoader(
            PoseDataset(track_dir, self.image_suffix, self.keypoint_suffix, self.pose_image_suffix),
            batch_size=self.batch_size, shuffle=True, pin_memory=True
        )
        for batch in tqdm(dataloader):
            images, pose_paths, pose_image_paths = batch
            if all([os.path.exists(p) for p in pose_image_paths]):
                print(f"Skip {pose_paths}")
                continue
            images = images.squeeze().cpu().numpy()
            result = self.model(images)
            pose_image = None
            if self.save_pose_image:
                pose_result, pose_image = result
            else:
                pose_result = result
            keypoints = pose_result[0]["keypoints"]
            np.savetxt(pose_paths[0], keypoints, fmt="%.6f")
            if self.save_pose_image:
                assert pose_image is not None
                cv2.imwrite(pose_image_paths[0], pose_image)


class PoseDataset(Dataset):
    def __init__(self, data_dir, image_suffix, keypoint_suffix, pose_image_suffix):
        self.all_image_paths = list(glob(f"{str(data_dir).rstrip('/')}/**/*{image_suffix}", recursive=True))
        self.all_pose_paths = [str(path).replace(image_suffix, keypoint_suffix) for path in self.all_image_paths]
        self.all_pose_image_paths = [
            str(path).replace(image_suffix, pose_image_suffix) for path in self.all_image_paths
        ]
        assert len(self.all_image_paths) == len(self.all_pose_paths) == len(self.all_pose_image_paths)

    def __len__(self):
        return len(self.all_image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.all_image_paths[idx]))
        return image, self.all_pose_paths[idx], self.all_pose_image_paths[idx]
