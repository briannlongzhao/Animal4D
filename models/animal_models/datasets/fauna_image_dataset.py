import os
from glob import glob
import random
from tqdm import tqdm
import re
from torch.utils.data import Dataset
import torchvision.datasets.folder
from torchvision.transforms.functional import InterpolationMode
from externals.Animals.model.dataset.util import *



class FaunaImageDataset(Dataset):
    def __init__(
        self, root, in_image_size=256, out_image_size=256, balance_category=True, load_dino_feature=False,
        load_dino_cluster=False, dino_feature_dim=16, load_keypoint=False, chunk_size=5000, load_in_chunks=True,
        shuffle=True, batch_size=8,
    ):
        super().__init__()
        self.root = root
        self.load_in_chunks = load_in_chunks
        self.categories = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.num_categories = len(self.categories)
        self.batch_size = batch_size
        self.in_image_size = in_image_size
        self.out_image_size = out_image_size
        self.image_loader = ["rgb.png", torchvision.datasets.folder.default_loader]
        self.mask_loader = ["mask.png", torchvision.datasets.folder.default_loader]
        self.bbox_loader = ["box.txt", box_loader]
        self.metadata_loader = ["metadata.json", metadata_loader]
        self.keypoint_loader = ["keypoint.txt", keypoint_loader]
        self.load_keypoint = load_keypoint
        self.image_transform = transforms.Compose(
            [transforms.Resize(self.out_image_size, interpolation=InterpolationMode.BILINEAR), transforms.ToTensor()])
        self.mask_transform = transforms.Compose(
            [transforms.Resize(self.out_image_size, interpolation=InterpolationMode.NEAREST), transforms.ToTensor()])
        self.load_dino_feature = load_dino_feature
        if load_dino_feature:
            self.dino_feature_loader = [f"feature.png", dino_loader, dino_feature_dim]
        self.load_dino_cluster = load_dino_cluster
        if load_dino_cluster:
            self.dino_cluster_loader = ["clusters.png", torchvision.datasets.folder.default_loader]
        self.category_to_paths = {}
        max_len = 0
        for category in self.categories:
            category_paths = self._parse_folder(os.path.join(root, category))
            self.category_to_paths[category] = category_paths
            max_len = max(max_len, len(category_paths))
        # Balance number of samples for all category
        if balance_category:
            for category, category_paths in self.category_to_paths.items():
                if len(category_paths) < max_len:
                    num_sample = max_len - len(category_paths)
                    category_paths = category_paths + random.choices(category_paths, k=num_sample)
                self.category_to_paths[category] = category_paths
                assert len(category_paths) == max_len
        # Split into batches and ensure no partial batch
        self.all_batches = []
        for cat, category_paths in list(self.category_to_paths.items()):
            if len(category_paths) % self.batch_size != 0:
                category_paths += random.choices(category_paths, k=batch_size - len(category_paths) % batch_size)
            if shuffle:
                random.shuffle(category_paths)
            self.category_to_paths[cat] = category_paths
            self.all_batches.extend([category_paths[i:i + batch_size] for i in range(0, len(category_paths), batch_size)])
        if shuffle:
            random.shuffle(self.all_batches)
        self.all_batches_flattened = [p for batch in self.all_batches for p in batch]
        if self.load_in_chunks:
            self.current_chunk_idx = None
            self.current_chunk = None
            self.chunk_size = chunk_size
            self.num_chunks = (len(self.all_batches_flattened) + self.chunk_size - 1) // self.chunk_size
            self.chunk_idx_to_samples = {}
            self.sample_idx_to_chunk_idx = {}
            for i in range(self.num_chunks):
                start = i * self.chunk_size
                end = min((i + 1) * self.chunk_size, len(self.all_batches_flattened))
                self.chunk_idx_to_samples[i] = self.all_batches_flattened[start:end]
                for j in range(start, end):
                    self.sample_idx_to_chunk_idx[j] = i

    def _load_ids(self, path, loader, transform=None):
        x = loader[1](path.format(loader[0]), *loader[2:])
        if transform:
            x = transform(x)
        return x

    def _parse_folder(self, path):
        image_path_suffix = self.image_loader[0]
        result = glob(os.path.join(path, '**/*' + image_path_suffix))
        # Only keep files with dino feature and keypoint
        result = [p for p in result if os.path.exists(p.replace(image_path_suffix, self.dino_feature_loader[0]))]
        result = [p for p in result if os.path.exists(p.replace(image_path_suffix, self.keypoint_loader[0]))]
        if '*' in image_path_suffix:
            image_path_suffix = re.findall(image_path_suffix, result[0])[0]
            self.image_loader[0] = image_path_suffix
        result = [p.replace(image_path_suffix, '{}') for p in result]
        return result

    def _load_chunk(self, index):
        chunk_idx = self.sample_idx_to_chunk_idx[index]
        if self.current_chunk_idx != chunk_idx:
            self.current_chunk_idx = chunk_idx
            self.current_chunk = []
            for path in tqdm(self.chunk_idx_to_samples[chunk_idx], desc=f"Loading chunk {chunk_idx}/{self.num_chunks}"):
                images = self._load_ids(path, self.image_loader, transform=self.image_transform).unsqueeze(0)
                masks = self._load_ids(path, self.mask_loader, transform=self.mask_transform).unsqueeze(0)
                mask_dt = compute_distance_transform(masks)
                metadata = self._load_ids(path, self.metadata_loader)
                global_frame_id = torch.LongTensor([int(metadata.get("clip_frame_id"))])
                xmin, ymin, xmax, ymax = metadata.get("crop_box_xyxy")
                full_w, full_h = metadata.get("video_frame_width", 1920), metadata.get("video_frame_height", 1080)
                bboxs = torch.Tensor(
                    [global_frame_id.item(), xmin, ymin, xmax - xmin, ymax - ymin, full_w, full_h, 0]).unsqueeze(0)
                # bboxs = self._load_ids(path, self.bbox_loader, transform=torch.FloatTensor).unsqueeze(0)
                # bboxs = torch.cat([bboxs, torch.Tensor([[category_idx]]).float()], dim=-1)  # pad a label number

                mask_valid = get_valid_mask(bboxs, (
                    self.out_image_size, self.out_image_size)
                                            )  # exclude pixels cropped outside the original image
                flows = None
                if self.load_dino_feature:
                    dino_features = self._load_ids(path, self.dino_feature_loader,
                                                   transform=torch.FloatTensor).unsqueeze(0)
                else:
                    dino_features = None
                if self.load_dino_cluster:
                    dino_clusters = self._load_ids(path, self.dino_cluster_loader,
                                                   transform=transforms.ToTensor()).unsqueeze(0)
                else:
                    dino_clusters = None
                if self.load_keypoint:
                    keypoint = self._load_ids(path, self.keypoint_loader).unsqueeze(0)
                    keypoint = keypoint / self.in_image_size * 2 - 1
                else:
                    keypoint = None

                seq_idx = torch.LongTensor([index])
                frame_idx = torch.LongTensor([0])

                bg_images = None

                out = (*map(none_to_nan, (
                    images, masks, mask_dt, mask_valid, flows, bboxs, bg_images, dino_features, dino_clusters, keypoint,
                    seq_idx, frame_idx
                )),)  # for batch collation
                self.current_chunk.append(out)

    def __len__(self):
        return len(self.all_batches_flattened)

    def __getitem__(self, index):
        self._load_chunk(index)
        return self.current_chunk[index % self.chunk_size]
