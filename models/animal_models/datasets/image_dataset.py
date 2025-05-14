import os
from glob import glob
import random
import re
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode

from externals.Animals.model.dataset.util import *


class ImageDataset(Dataset):
    def __init__(
        self, root, in_image_size=256, out_image_size=256, random_xflip=False, load_dino_feature=False,
        load_dino_cluster=False, dino_feature_dim=16, load_depth=False, load_articulation=False,
        reverse_articulation=False, load_keypoint=True, chunk_size=10000, shuffle=True, load_in_chunks=True,
    ):
        super().__init__()
        self.image_loader = ["rgb.png", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.bbox_loader = ["box.txt", box_loader]
        if load_dino_feature:
            self.dino_feature_loader = [f"feature.png", dino_loader, dino_feature_dim]
        self.samples = self._parse_folder(root)
        self.shuffle = shuffle
        self.load_in_chunks = load_in_chunks
        if self.shuffle:
            random.shuffle(self.samples)
        else:
            self.samples.sort()
        if self.load_in_chunks:
            self.current_chunk_idx = None
            self.current_chunk = None
            self.chunk_size = chunk_size
            self.num_chunks = (len(self.samples) + self.chunk_size - 1) // self.chunk_size
            self.chunk_idx_to_samples = {}
            self.sample_idx_to_chunk_idx = {}
            for i in range(self.num_chunks):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, len(self.samples))
                self.chunk_idx_to_samples[i] = self.samples[start:end]
                for j in range(start, end):
                    self.sample_idx_to_chunk_idx[j] = i
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        self.load_dino_feature = load_dino_feature
        self.depth_transform = transforms.Compose([transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor(), transforms.Lambda(lambda x: x.float())])
        self.load_depth = load_depth
        self.load_articulation = load_articulation
        self.load_keypoint = load_keypoint
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        if load_depth:
            self.depth_loader = ["depth.png", depth_loader]
        if load_articulation:
            self.articulation_loader = ["articulation.txt", articulation_loader]
        if load_keypoint:
            self.keypoint_loader = ["keypoint.txt", keypoint_loader]
        self.reverse_articulation = reverse_articulation
        self.random_xflip = random_xflip
        if self.load_articulation and self.reverse_articulation:
            self.reverse_idx_pairs = [
                [0, 4], [1, 5], [2, 6], [3, 7],  # Spine
                [8, 14], [9, 15], [10, 16],  # LF <-> RR
                [11, 17], [12, 18], [13, 19],  # RF <-> LR
            ]

    def _parse_folder(self, path):  # TODO: currently only support one image suffix
        image_path_suffix = self.image_loader[0]
        result = glob(os.path.join(path, '**/*'+image_path_suffix), recursive=True)
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]
        return result

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def _load_chunk(self, index):
        chunk_idx = self.sample_idx_to_chunk_idx[index]
        if self.current_chunk_idx != chunk_idx:
            self.current_chunk_idx = chunk_idx
            self.current_chunk = []
            for path in tqdm(self.chunk_idx_to_samples[chunk_idx], desc=f"Loading chunk {chunk_idx}/{self.num_chunks}"):
                images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)
                masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
                mask_dt = compute_distance_transform(masks)
                bboxs = self._load_ids(path, self.bbox_loader, transform=torch.FloatTensor).unsqueeze(0)
                mask_valid = get_valid_mask(bboxs, (self.out_image_size, self.out_image_size))
                flows = None
                bg_images = None
                if self.load_dino_feature:
                    dino_features = self._load_ids(
                        path, self.dino_feature_loader, transform=torch.FloatTensor
                    ).unsqueeze(0)
                else:
                    dino_features = None
                if self.load_dino_cluster:
                    dino_clusters = self._load_ids(
                        path, self.dino_cluster_loader, transform=transforms.ToTensor()).unsqueeze(0)
                else:
                    dino_clusters = None
                if self.load_depth:
                    depth = self._load_ids(path, self.depth_loader, transform=self.depth_transform).unsqueeze(0)
                    depth = normalize_depth(depth, masks[:, [0]])
                else:
                    depth = None
                if self.load_keypoint:
                    keypoint = self._load_ids(path, self.keypoint_loader).unsqueeze(0)
                    keypoint = keypoint / self.in_image_size * 2 - 1
                else:
                    keypoint = None
                if self.load_articulation:
                    articulation, articulation_flag = self._load_ids(path, self.articulation_loader)
                    if articulation is not None:
                        if self.reverse_articulation:
                            articulation_reversed = torch.zeros_like(articulation)
                            for idx1, idx2 in self.reverse_idx_pairs:
                                articulation_reversed[idx1] = -articulation[idx2]
                                articulation_reversed[idx2] = -articulation[idx1]
                            articulation_reversed[:, 2] = -articulation_reversed[:, 2]  # Do not reverse dimension 2
                            articulation = articulation_reversed
                        articulation = articulation.unsqueeze(0)
                else:
                    articulation, articulation_flag = None, None
                seq_idx = torch.LongTensor([index])
                frame_idx = torch.LongTensor([0])

                ## random horizontal flip
                if self.random_xflip and np.random.rand() < 0.5:
                    xflip = lambda x: None if x is None else x.flip(-1)
                    images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters = (
                    *map(xflip, (images, masks, mask_dt, mask_valid, flows, bg_images, dino_features, dino_clusters)),)
                    bboxs = horizontal_flip_box(bboxs)  # NxK
                out = (*map(none_to_nan, (
                images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint,
                seq_idx, frame_idx)),)
                self.current_chunk.append(out)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        self._load_chunk(index)
        return self.current_chunk[index % self.chunk_size]


def normalize_depth(depth, mask=None):
    # Normalize depth values to [0, 1] range based on the minimum and maximum depth values in masked regions
    # Set all background to 0
    if mask is not None:
        depth_for_min = torch.where(mask.bool(), depth, torch.full_like(depth, float('inf')))
        depth_for_max = torch.where(mask.bool(), depth, torch.full_like(depth, -float('inf')))
    else:
        depth_for_min = depth
        depth_for_max = depth
    depth_min = depth_for_min.amin(dim=(-1, -2), keepdim=True)
    depth_max = depth_for_max.amax(dim=(-1, -2), keepdim=True)
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    if mask is not None:
        normalized_depth = torch.where(mask.bool(), normalized_depth, torch.zeros_like(normalized_depth))
    return normalized_depth
