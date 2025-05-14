import os
import random
import time
from glob import glob
import torch
from torchvision import transforms
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import Dataset, Subset
import torch.nn.functional as F
import numpy as np
import faiss

from models.utils import Profiler, gpu_memory_usage
from models.feature_models import DINOExtractor, Denoiser


class FeatureExtractor:
    def __init__(
        self, dataset_dir, model_type, stride, image_size, image_pad, denoise, pca_dim, feature_batch_size,
        pca_fit_samples, image_suffix, mask_suffix, feature_suffix, pca_mode, seed, device, pca_path=None
    ):
        self.dataset_dir = Path(dataset_dir)
        self.denoise = denoise
        self.image_pad = image_pad
        self.feature_batch_size = feature_batch_size
        self.stride = stride
        self.image_size = image_size
        self.model_type = model_type
        self.pca_dim = pca_dim
        self.pca_fit_samples = pca_fit_samples
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix
        self.feature_suffix = feature_suffix
        self.pca_mode = pca_mode
        self.device = device
        self.pca_path = Path(pca_path)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        if "dino" in self.model_type:
            self.extractor = DINOExtractor(model_type=self.model_type, stride=stride, device=device)
        else:
            raise NotImplementedError
        self.feature_dim = self.extractor.model.embed_dim
        if self.denoise:
            pos_h = 37 if "dinov2" in self.model_type else 32
            self.denoiser = Denoiser(
                noise_map_height=pos_h, noise_map_width=pos_h, feature_dim=self.feature_dim,
                enable_pe=True, denoiser_type="transformer",
            )
            if self.feature_dim == 768:
                ckpt = torch.load("externals/Denoising_ViT/ckpt/vit_base_patch14_dinov2.lvd142m.pth.1")["denoiser"]
                missing, unexpected = self.denoiser.load_state_dict(ckpt, strict=False)
                self.denoiser.eval()
                self.denoiser.to(self.device)
                print("Missing keys:", missing)
                print("Unexpected keys:", unexpected)
            else:
                self.denoise = False
        else:
            self.denoiser = None

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor((0.485, 0.456, 0.406)),
                std=torch.tensor((0.229, 0.224, 0.225)))
        ])
        self.transform_no_norm = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

        print("model_type:", self.model_type, flush=True)
        print("denoise:", self.denoise, flush=True)
        print("image_size:", self.image_size, flush=True)
        print("stride:", self.stride, flush=True)
        print("model_feature_dim:", self.extractor.model.embed_dim, flush=True)

    def create_dataloader(self, root, num_samples=None, shuffle=False):
        dataset = FeatureExtractionDataset(
            data_dir=root, image_suffix=self.image_suffix, mask_suffix=self.mask_suffix, transform=self.transform,
            transform_no_norm=self.transform_no_norm, mask_transform=self.mask_transform
        )
        if num_samples and num_samples < len(dataset):
            dataset = Subset(dataset, np.random.choice(len(dataset), num_samples, replace=False))
        return DataLoader(dataset, batch_size=self.feature_batch_size, shuffle=shuffle, pin_memory=True, num_workers=0)

    def extract_features(self, image):
        """Extract features from a batch of image with shape (b, c, h, w)"""
        image_padded = torch.nn.functional.pad(image, 4 * [self.image_pad], mode="reflect")
        with torch.no_grad():
            features = self.extractor(image_padded)
            if self.denoise:
                features = rearrange(
                    features, "b 1 (h w) d -> b h w d",
                    h=self.extractor.num_patches[0], w=self.extractor.num_patches[1]
                )
                features = self.denoiser(features)
                features = rearrange(features, "b h w d -> b 1 (h w) d")
            features = F.normalize(features, p=2, dim=-1)
        return features

    def fit_pca(self, data_root):
        dataloader = self.create_dataloader(data_root, shuffle=True, num_samples=self.pca_fit_samples)
        all_features, features_batch, masks_batch = [], [], []
        num_features = 0
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            image, image_path, mask = batch["image"], batch["image_path"], batch["mask"]
            features = self.extract_features(image)
            mask = torch.nn.functional.interpolate(batch["mask"], size=self.extractor.num_patches)
            features_batch.append(features)
            masks_batch.append(mask)
            num_features += features.shape[0]
            if gpu_memory_usage() > 0.8 or (idx + 1) == len(dataloader):  # Move to CPU periodically
                features_batch = torch.cat(features_batch)
                masks_batch = torch.cat(masks_batch).bool()
                features_batch = rearrange(features_batch, "b 1 t d -> (b t) d")
                masks_batch = rearrange(masks_batch, "b 1 h w -> (b h w)")
                features_batch = features_batch[masks_batch].cpu().numpy().astype(np.float16)
                all_features.append(features_batch)
                features_batch, masks_batch = [], []
                num_features = 0
            torch.cuda.empty_cache()
            allocated_memory = torch.cuda.memory_allocated()
            reserved_memory = torch.cuda.memory_reserved()
            print(f"\nAllocated: {allocated_memory / (1024 ** 2):.2f} MB, Reserved: {reserved_memory / (1024 ** 2):.2f} MB")
        all_features = np.concatenate(all_features, axis=0)
        pca_mat = faiss.PCAMatrix(all_features.shape[1], self.pca_dim)
        pca_mat.train(all_features)
        assert pca_mat.is_trained
        return pca_mat

    def save_features(self, features, image_paths, save_dir=None):
        assert len(image_paths) == len(features), \
            f"Number of images {len(image_paths)} and features {len(features)} mismatch"
        save_paths = [image_path.replace(self.image_suffix, self.feature_suffix) for image_path in image_paths]
        if save_dir is not None:
            save_paths = [os.path.join(save_dir, os.path.basename(save_path)) for save_path in save_paths]
            os.makedirs(save_dir, exist_ok=True)
        mins = features.min(axis=(1, 2), keepdims=True)
        maxs = features.max(axis=(1, 2), keepdims=True)
        features = (features - mins) / (maxs - mins)
        for save_path, feature in zip(save_paths, features):
            if os.path.exists(save_path):
                continue
            print(f"Saving feature to {save_path}")
            assert feature.min() >= 0 and feature.max() <= 1
            # feat = np.round(((feat + 1) * 127)).astype('uint8')
            feature = np.round(feature * 255).astype('uint8')
            # append channels to make it divisible by 3
            n_channels = feature.shape[2]
            n_addon_channels = int(np.ceil(n_channels / 3) * 3) - n_channels
            feat = np.concatenate(
                [feature, np.zeros([feature.shape[0], feature.shape[0], n_addon_channels], dtype=feature.dtype)],
                axis=-1
            )
            feature = rearrange(feat, 'h w (t c) -> h (t w) c', c=3)
            Image.fromarray(feature).save(save_path)

    def apply_pca(self, data_root, pca_mat):
        dataloader = self.create_dataloader(data_root, shuffle=True)
        features_batch, image_path_batch = [], []
        num_features = 0
        for idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
            if self.skip_batch(batch) and (idx + 1) != len(dataloader):
                continue
            features = self.extract_features(batch["image"])
            num_features += features.shape[0]
            features = rearrange(features, "b 1 t d -> (b t) d")
            features_batch.append(features)
            image_path_batch.extend(batch["image_path"])
            if gpu_memory_usage() > 0.5 or (idx + 1) == len(dataloader):  # Move to CPU periodically
                features = torch.cat(features_batch).cpu().numpy()
                features = pca_mat.apply_py(features)
                features = rearrange(
                    features, "(b h w) d -> b h w d",
                    b=num_features, h=self.extractor.num_patches[0], w=self.extractor.num_patches[1]
                )
                self.save_features(features=features, image_paths=image_path_batch)
                features_batch, image_path_batch = [], []
                num_features = 0
                torch.cuda.empty_cache()

    def run(self):
        if self.pca_mode == "fit":
            train_root = self.dataset_dir / "train"
            assert train_root.is_dir(), f"Train root {train_root} does not exist"
            with Profiler("fit", num_frames=self.pca_fit_samples):
                pca_mat = self.fit_pca(train_root)
            faiss.write_VectorTransform(pca_mat, str(self.pca_path))
            print(f"PCA matrix saved to {self.pca_path}")
        elif self.pca_mode == "apply":
            pca_mat = faiss.read_VectorTransform(str(self.pca_path))
            print("PCA matrix dim:", pca_mat.d_in, pca_mat.d_out)
            self.apply_pca(self.dataset_dir, pca_mat)
        else:
            raise NotImplementedError
        return

    # def all_processed(self, data_path): Deprecated
    #     """Check if all images have been processed if not using database"""
    #     rgb_paths = list(Path(data_path).rglob('*' + self.image_suffix))
    #     feature_paths = [str(rgb_path).replace(self.image_suffix, self.feature_suffix) for rgb_path in rgb_paths]
    #     if not all([Path(feature_path).exists() for feature_path in feature_paths]):
    #         return False
    #     return True

    def skip_batch(self, batch):
        for image_path in batch["image_path"]:
            feature_path = image_path.replace(self.image_suffix, self.feature_suffix)
            if not os.path.exists(feature_path):
                return False
        return True


class FeatureExtractionDataset(Dataset):
    def __init__(
        self, data_dir, image_suffix, mask_suffix, transform=None, transform_no_norm=None,
        mask_transform=None, load_mask=True
    ):
        self.image_paths = glob(os.path.join(data_dir, '**/*' + image_suffix), recursive=True)
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.transform_no_norm = transform_no_norm if transform_no_norm is not None else transforms.ToTensor()
        self.mask_transform = mask_transform if mask_transform is not None else transforms.ToTensor()
        self.load_mask = load_mask
        self.image_suffix = image_suffix
        self.mask_suffix = mask_suffix

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = str(self.image_paths[idx])
            image = Image.open(image_path)
            image_transform = self.transform(image)
            image_transform_no_norm = self.transform_no_norm(image)
            result = {"image": image_transform, "image_no_norm": image_transform_no_norm, "image_path": image_path}
            if self.load_mask:
                mask_path = image_path.replace(self.image_suffix, self.mask_suffix)
                mask = Image.open(mask_path)
                mask_transformed = self.mask_transform(mask)[:1]
                result["mask"] = mask_transformed
        except OSError:
            print(f"Failed to load image {image_path}")
            raise Exception
        return result
