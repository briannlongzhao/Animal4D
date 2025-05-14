import os
import cv2
import math
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from models.depth_models import DepthAnything, DepthAnythingV2
from models.utils import get_frame_images

"""
Mostly adapted from 
https://github.com/HusamJubran/video_object_processing/blob/main/scripts/preprocess_1st_stage_per_video.py
"""


class OcclusionProcessor:  # TODO: batchify all operations if possible
    """Module to process depth maps and occlusion boundary maps"""
    def __init__(self, depth_method, occlusion_batch_size, depth_suffix, mask_suffix, occlusion_suffix, device):
        self.depth_method = depth_method
        self.batch_size = occlusion_batch_size
        self.depth_suffix = depth_suffix
        self.mask_suffix = mask_suffix
        self.occlusion_suffix = occlusion_suffix
        self.edge_pool_size = 5
        self.device = device

        if self.depth_method == "patchfusion":  # TODO: deprecate or see how to improve speed
            self.depth_mapper = PatchFusion(device=self.device)
            self.batch_size = 1
        elif self.depth_method == "depth_anything":
            self.depth_mapper = DepthAnything(device=self.device)
        elif self.depth_method == "depth_anything_v2":
            self.depth_mapper = DepthAnythingV2(device=self.device)
        else:
            raise NotImplementedError

    @staticmethod
    def detect_borders_binary_mask(binary_mask, blur_kernel_size, dilation_kernel_size):
        """
        Detects and dilates the borders within a binary mask using average pooling and dilation techniques.

        Args:
        binary_mask (Tensor): Input binary mask where non-zero (true) areas are considered the 'object'.
        blur_kernel_size (int): The size of the kernel used for average pooling, affecting edge detection sensitivity.
        dilation_kernel_size (int): The size of the kernel used for dilation, affecting the thickness of edges.

        Returns:
        tuple: A tuple containing the original binary mask in uint8 format and the dilated edge map.
        """
        # Convert the input mask to uint8 format with values as 0 or 255
        mask = (binary_mask * 255).float()

        # Apply average pooling to blur the image, which helps in highlighting the borders on subtraction
        blur_kernel_size = blur_kernel_size + 1 if blur_kernel_size % 2 == 0 else blur_kernel_size
        pooled_mask = F.avg_pool2d(mask, kernel_size=blur_kernel_size, stride=1, padding=blur_kernel_size // 2)

        # Compute the edge map by finding the difference between the original mask and its blurred version
        edge_map = (mask - pooled_mask).abs()

        # Apply a threshold to isolate the edges; edges are expected to have high differences
        edge_map = (edge_map > 50).float()

        # Apply dilation to thicken the edges, making them more visible and defined
        # kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        # edge_map_dilated = cv2.dilate(edge_map_normalized, kernel, iterations=1)

        dilation_kernel_size = dilation_kernel_size + 1 if dilation_kernel_size % 2 == 0 else dilation_kernel_size
        edge_map_dilated = F.max_pool2d(edge_map, kernel_size=3, stride=1, padding=3 // 2)
        edge_map_dilated = (edge_map_dilated > 0).to(torch.uint8)

        return edge_map_dilated

    @staticmethod
    def shrink_mask(mask, pixels=1):
        """
        Shrinks the given mask by averaging over local neighborhoods, reducing edge influence.

        Args:
        mask (numpy array): The original binary mask.
        pixels (int): Defines the size of the neighborhood used for shrinking.

        Returns:
        numpy array: The shrunk mask as a numpy array, still in binary format.
        """
        # Convert the numpy mask to a PyTorch tensor and add required dimensions
        # mask_tensor = torch.FloatTensor(mask)[None, None, :, :]

        # Set the kernel size for average pooling based on the number of pixels to shrink by
        kernel_size = pixels * 2 + 1

        # Perform average pooling to reduce the mask size; adjust stride and padding to maintain original size
        averaged_mask = torch.nn.functional.avg_pool2d(mask.float(), kernel_size, stride=1, padding=pixels)

        # Threshold the averaged results to maintain a binary mask
        shrunk_mask = (averaged_mask > 0.9999).float()

        # Convert the tensor back to a numpy array for further processing
        return shrunk_mask

    def apply_mask_and_fill(self, image, mask):
        """
        Applies a mask to an image and fills in the zeroed values with the nearest non-zero pixel value.
        """
        # Apply the mask
        masked_image = image * mask

        # Compute the distance transform
        # distances, (indices_x, indices_y) = distance_transform_edt(mask == 0, return_indices=True)
        # TODO: look for batched edt
        indices_x, indices_y = [], []
        for m in range(masked_image.shape[0]):
            _, (index_x, index_y) = distance_transform_edt(mask[m].cpu().numpy() == 0, return_indices=True)
            indices_x.append(index_x)
            indices_y.append(index_y)

        # Prepare an empty image to fill
        indices_x = torch.tensor(np.array(indices_x), device=masked_image.device)
        indices_y = torch.tensor(np.array(indices_y), device=masked_image.device)
        batch_indices = torch.arange(masked_image.size(0), device=masked_image.device).view(-1, 1, 1)
        filled_image = masked_image[batch_indices, indices_x, indices_y]

        #filled_image = masked_image[indices_x, indices_y]

        return filled_image, masked_image

    @staticmethod
    def convert_occ_to_rgb(border_depth, border, high_threshold, low_threshold):
        """
        Converts a depth map at specified border areas into an RGB image for indicating occluded, maybe occluded and occluded,
        marking different depth ranges with specific colors and calculates the occluded proportion.

        Args:
        border_depth (array): 2D array containing depth values at border areas.
        border (array): Binary mask where 1 represents border pixels to analyze.
        high_threshold (float): Above this value it is not occluded.
        low_threshold (float): Below this value it is occluded.

        Returns:
        tuple: A tuple containing the RGB image (numpy array) and the occluded proportion (float).
        """
        rgb_image = torch.zeros((*border_depth.shape, 3), dtype=torch.float32, device=border_depth.device)
        border_mask = (border == 1)
        not_occluded_mask = (border_depth > high_threshold) & border_mask
        uncertain_mask = (border_depth > low_threshold) & (border_depth <= high_threshold) & border_mask
        occluded_mask = (border_depth <= low_threshold) & border_mask

        # Assign colors based on masks
        rgb_image[not_occluded_mask] = torch.tensor([0, 1, 0], dtype=torch.float32, device=rgb_image.device)  # Green for not occluded
        rgb_image[uncertain_mask] = torch.tensor([0, 0, 1], dtype=torch.float32, device=rgb_image.device)  # Blue for uncertain
        rgb_image[occluded_mask] = torch.tensor([1, 0, 0], dtype=torch.float32, device=rgb_image.device)  # Red for occluded
        rgb_image = (rgb_image * 255).to(torch.uint8)

        # Calculate counts
        unoccluded_counter = torch.sum(not_occluded_mask, dim=(-1, -2))
        uncertain_counter = torch.sum(uncertain_mask, dim=(-1, -2))
        occluded_counter = torch.sum(occluded_mask, dim=(-1, -2))

        # Calculate occluded proportion
        occluded = uncertain_counter + occluded_counter
        occluded_proportion = torch.where(
            (occluded + unoccluded_counter) > 0,
            occluded / (occluded + unoccluded_counter),
            torch.zeros_like(occluded, device=occluded.device)
        )
        return rgb_image, occluded_proportion

    def calculate_occlusion(
        self, depth_map, mask, border_pool_size=5, occlusion_high_threshold=0.02, occlusion_low_threshold=-0.01
    ):  # Thresholds may need to be changed based on the depth method used
        """
        Calculates occlusion based on the depth map differences between object and background,
        using depth thresholds to determine occluded areas and providing a visualization.
        """
        object_area = torch.sum(mask, dim=(-1,-2))
        object_area_sqrt = torch.sqrt(object_area)
        border_dilation_kernel_size = int(torch.round(object_area_sqrt.mean() / 85).item())
        border = self.detect_borders_binary_mask(
            mask, blur_kernel_size=border_pool_size, dilation_kernel_size=border_dilation_kernel_size
        )
        shrinked_object = self.shrink_mask(mask, border_dilation_kernel_size)
        min_depth = torch.min(depth_map)
        max_depth = torch.max(depth_map)
        depth_map = (depth_map - min_depth) / (max_depth - min_depth)
        object_filled_depth_map, object_masked_depth_map = self.apply_mask_and_fill(depth_map, shrinked_object)

        # Shrink the background mask and fill the depth map within the shrunk background area
        shrinked_background = self.shrink_mask(1 - mask, border_dilation_kernel_size)
        background_filled_depth_map, background_masked_depth_map = self.apply_mask_and_fill(depth_map, shrinked_background)

        # Calculate depth maps for object and background borders
        object_border_depth = border * object_filled_depth_map
        background_border_depth = border * background_filled_depth_map

        # Calculate the depth difference between background and object at the borders
        border_depth_diff = object_border_depth - background_border_depth

        # Convert depth difference to RGB visualization and calculate occlusion proportion
        occlusion_rgb, occluded_proportion = self.convert_occ_to_rgb(
            border_depth=border_depth_diff, border=border, high_threshold=occlusion_high_threshold,
            low_threshold=occlusion_low_threshold
        )
        return occlusion_rgb, occluded_proportion

    def extract_depths(self, tracking_results, video_path, frames_dir, depths_dir):
        os.makedirs(depths_dir, exist_ok=True)
        frame_ids = [k for k, v in tracking_results.items() if len(v) > 0]
        frame_ids = list(range(min(frame_ids), max(frame_ids) + 1)) if len(frame_ids) > 0 else []
        dataloader = DataLoader(
            DepthDataset(video_path, frames_dir, frame_ids=frame_ids),
            collate_fn=lambda batch: (list(zip(*batch))[0], list(zip(*batch))[1]),
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        for images, frame_ids in dataloader:
            depths = self.depth_mapper(images)
            depth_filenames = [os.path.join(depths_dir, f"{str(frame).zfill(8)}.png") for frame in frame_ids]
            depths = depths.detach()
            depths = 65535 * (
                (depths - torch.amin(depths, dim=(1, 2), keepdim=True)) / (torch.amax(depths, dim=(1, 2), keepdim=True)
                - torch.amin(depths, dim=(1, 2),keepdim=True))
            )
            depths = depths.cpu().numpy().astype(np.uint16)
            for i, filename in enumerate(depth_filenames):
                cv2.imwrite(filename, depths[i])
        return

    def run_track(self, track_dir):
        dataloader = DataLoader(
            OcclusionDataset(track_dir, self.depth_suffix, self.mask_suffix),
            batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        frame_id_to_occlusion = {}
        for batch in dataloader:
            depth = batch["depth"].to(device=self.device, dtype=torch.int32)
            mask, path = batch["mask"].to(self.device), batch["path"]
            occlusion_images, occlusions = self.calculate_occlusion(depth, mask)
            occlusion_images = occlusion_images.cpu().numpy()
            occlusion = occlusions.tolist()
            for i in range(len(occlusions)):
                occlusion_path = batch["path"][i].replace(self.depth_suffix, self.occlusion_suffix)
                occlusion_image = cv2.cvtColor(occlusion_images[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(occlusion_path, occlusion_image)
                frame_id = int(os.path.basename(occlusion_path).split("_")[0])
                frame_id_to_occlusion[frame_id] = occlusion[i]
        return frame_id_to_occlusion


class DepthDataset(Dataset):  # TODO: maybe load all data in __init__
    def __init__(self, video_path, frames_dir, frame_ids):
        self.frame_ids = frame_ids
        self.images = get_frame_images(video_path, frames_dir, frame_ids)
        assert len(self.frame_ids) == len(self.images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.frame_ids[idx]


class OcclusionDataset(Dataset):  # TODO: maybe load all data in __init__
    def __init__(self, data_dir, depth_suffix, mask_suffix):
        self.all_depth_paths = list(Path(data_dir).rglob(f"*{depth_suffix}"))
        self.all_mask_paths = [Path(str(path).replace(depth_suffix, mask_suffix)) for path in self.all_depth_paths]

    def __len__(self):
        return len(self.all_depth_paths)

    def __getitem__(self, idx):
        depth_path = self.all_depth_paths[idx]
        depth = np.array(Image.open(str(depth_path)))
        mask_path = self.all_mask_paths[idx]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)
        data = {
            "depth": depth,
            "mask": mask,
            "path": str(depth_path)
        }
        return data
