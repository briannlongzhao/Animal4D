import os
import cv2
import torch
from bisect import insort
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from models.flow_models import SEA_RAFT


class FlowProcessor:
    def __init__(self, flow_method, image_suffix, flow_suffix, mask_suffix, flow_batch_size, device):
        self.image_suffix = image_suffix
        self.flow_suffix = flow_suffix
        self.mask_suffix = mask_suffix
        self.batch_size = flow_batch_size
        self.device = device
        if flow_method == "sea_raft":
            self.model = SEA_RAFT(device)
        else:
            raise NotImplementedError

    def run_track(self, track_dir):
        dataloader = DataLoader(
            FlowDataset(track_dir, self.image_suffix, self.mask_suffix),
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        frame_id_to_flow = {}
        for batch in dataloader:
            image1, image2 = batch["image1"], batch["image2"]
            flows, flow_images = self.model(image1, image2)
            for i in range(len(flows)):
                image_path = batch["path1"][i]
                flow_path = image_path.replace(self.image_suffix, self.flow_suffix)
                flow_image = cv2.cvtColor(flow_images[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(flow_path, flow_image)
                frame_id = int(os.path.basename(image_path).split("_")[0])
                mask1 = batch["mask1"][i].to(self.device)
                masked_flow = torch.norm(flows[i] * mask1, p=2, dim=0)
                masked_flow = (masked_flow.sum() / torch.count_nonzero(mask1)).item()
                frame_id_to_flow[frame_id] = masked_flow
        return frame_id_to_flow



class FlowDataset(Dataset):  # TODO: maybe load all data in __init__
    def __init__(self, data_dir, image_suffix, mask_suffix=None):
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.all_image_pairs = []
        data_dir = Path(data_dir)
        track_dir_to_frame_paths = {}
        all_image_paths = data_dir.rglob(f"*{image_suffix}")
        for image_path in all_image_paths:
            track_dir = str(image_path.parent)
            if track_dir not in track_dir_to_frame_paths.keys():
                track_dir_to_frame_paths[track_dir] = []
            frame_path = image_path.name
            insort(track_dir_to_frame_paths[track_dir], frame_path)
        for track_dir, frame_paths in track_dir_to_frame_paths.items():
            for i in range(1, len(frame_paths)):
                image_path1 = str(Path(track_dir) / frame_paths[i-1])
                image_path2 = str(Path(track_dir) / frame_paths[i])
                self.all_image_pairs.append((image_path1, image_path2))

    def __len__(self):
        return len(self.all_image_pairs)

    def __getitem__(self, idx):
        image_path1, image_path2 = self.all_image_pairs[idx]
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image_tensor1 = torch.tensor(image1, dtype=torch.float32).permute(2, 0, 1)
        image_tensor2 = torch.tensor(image2, dtype=torch.float32).permute(2, 0, 1)
        mask_1 = cv2.imread(image_path1.replace(self.image_suffix, self.mask_suffix), cv2.IMREAD_GRAYSCALE)
        mask_2 = cv2.imread(image_path2.replace(self.image_suffix, self.mask_suffix), cv2.IMREAD_GRAYSCALE)
        mask_tensor1 = torch.tensor(mask_1 > 0)
        mask_tensor2 = torch.tensor(mask_2 > 0)
        data = {
            "image1": image_tensor1,
            "image2": image_tensor2,
            "mask1": mask_tensor1,
            "mask2": mask_tensor2,
            "path1": image_path1,
            "path2": image_path2,
        }

        return data
