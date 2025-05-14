import os
import sys
import math
import logging
from pathlib import Path

import configargparse
import faiss
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from einops import rearrange
from detectron2.data import transforms as T

from externals.extract_dino.extractor import ViTExtractor
from externals.extract_dino.feature_storage import save_feat_as_img
sys.path.append("externals/Denoising_ViT")
from externals.Denoising_ViT.DenoisingViT import Denoiser


logger = logging.getLogger("logger")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%I:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)


def create_dataloader(
    root, batch_size, img_postfix, mask_postfix, shuffle=False, transform=None, transform_no_norm=None,
    mask_transform=None
):
    img_names = sorted(list(Path(root).rglob('*' + img_postfix)))
    if mask_postfix is not None:
        mask_names = sorted(list(Path(root).rglob('*' + mask_postfix)))
        assert len(img_names) == len(mask_names)
    else:
        mask_names = None
    dataset = FilesListDataset(
        img_names, masks_paths=mask_names, transform=transform, transform_no_norm=transform_no_norm,
        mask_transform=mask_transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def save_features(img_paths, features, img_postfix, feat_postfix, save_dir=None):
    assert len(img_paths) == len(features)
    save_paths = [img_path.replace(img_postfix, feat_postfix) for img_path in img_paths]
    if save_dir is not None:
        save_paths = [os.path.join(save_dir, os.path.basename(save_path)) for save_path in save_paths]
        os.makedirs(save_dir, exist_ok=True)
    mins = features.min(axis=(1, 2), keepdims=True)
    maxs = features.max(axis=(1, 2), keepdims=True)
    features = (features - mins) / (maxs - mins)
    for save_path, feature in zip(save_paths, features):
        save_feat_as_img(save_path, feature.clip(-1, 1))


def extract_features(args, extractor, dataloader, denoiser=None):
    with torch.no_grad():
        all_features = []
        all_masks = []
        all_paths = []
        for batch in tqdm(dataloader, total=len(dataloader)):
            img, img_no_norm, img_path = (batch[k] for k in ['img', 'img_no_norm', 'img_path'])
            img = img.to(extractor.device)
            img_padded = torch.nn.functional.pad(img, 4 * [args.img_pad], mode='reflect')
            if "dinov2" in args.model_type:
                extractor.extract_descriptors(
                    img_padded, facet=args.facet, layer=args.layer, bin=False
                )
                features = extractor.model.forward_features(img_padded)['x_norm_patchtokens'].unsqueeze(1)
            elif "dino" in args.model_type:
                features = extractor.extract_descriptors(
                    img_padded, facet=args.facet, layer=args.layer, bin=False
                )
            else:
                raise NotImplementedError
            if args.denoise:
                assert denoiser is not None
                features = rearrange(
                    features, "b 1 (h w) d -> b h w d", h=extractor.num_patches[0], w=extractor.num_patches[1]
                )
                features = denoiser(features)
                features = rearrange(features, "b h w d -> b 1 (h w) d")
            features = F.normalize(features, p=2, dim=-1)
            features = features.cpu().numpy()
            all_features.append(features)
            all_paths += img_path
            if args.load_mask:
                mask = batch['mask']
                mask = torch.nn.functional.interpolate(mask, size=extractor.num_patches).cpu().numpy()
                all_masks.append(mask)
        all_features = np.concatenate(all_features, axis=0)
        if args.load_mask:
            all_masks = np.concatenate(all_masks, axis=0)
    return all_features, all_masks, all_paths


def fit_pca(features, out_dim, masks=None, random_sample=None):
    n, _, t, _ = features.shape
    features = rearrange(features, "n 1 t d -> (n t) d")
    if masks is not None and len(masks) > 0:
        features = features[masks.reshape(-1, 1).astype(np.bool).squeeze()]
    if random_sample is not None:
        num_samples = min(random_sample * t, features.shape[0])
        random_indices = np.random.choice(features.shape[0], num_samples, replace=False)
        features = features[random_indices, :]
    pca_mat = faiss.PCAMatrix(features.shape[1], out_dim)
    pca_mat.train(features)
    assert pca_mat.is_trained
    return pca_mat





def main(args):
    print("model_type:", args.model_type, flush=True)
    print("denoise:", args.denoise, flush=True)
    print("image_size:", args.image_size, flush=True)
    print("stride:", args.stride, flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pca_out_path = Path(args.results_info_root) / 'pca.faiss'
    feat_postfix = f"_{args.model_type}_{args.pca_dim}.png"
    extractor = ViTExtractor(model_type=args.model_type, stride=args.stride, model_path=args.model_path, device=device)
    if args.denoise:  # TODO: update for official checkpoint or implement pe interpolation
        pos_h = 37 if "dinov2" in args.model_type else 32
        denoiser = Denoiser(
            noise_map_height=pos_h, noise_map_width=pos_h, feature_dim=extractor.model.embed_dim, vit=None,
            enable_pe=True, denoiser_type="transformer",
        )
        ckpt = torch.load(args.denoiser_ckpt_path)["denoiser"]
        missing, unexpected = denoiser.load_state_dict(ckpt, strict=False)
        denoiser.eval()
        denoiser.to(device)
        print("Missing", missing)
        print("Unexpected", unexpected)
    else:
        denoiser = None
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor((0.485, 0.456, 0.406)),
            std=torch.tensor((0.229, 0.224, 0.225)))
    ])
    transform_no_norm = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    train_dataloader = create_dataloader(
        args.train_root, args.batch_size, args.img_postfix, args.mask_postfix, shuffle=args.shuffle_data,
        transform=transform, transform_no_norm=transform_no_norm, mask_transform=mask_transform
    )
    test_dataloader = create_dataloader(
        args.test_root, args.batch_size, args.img_postfix, args.mask_postfix, shuffle=args.shuffle_data,
        transform=transform, transform_no_norm=transform_no_norm, mask_transform=mask_transform
    )

    logger.info("Extracting train features")
    features_train, masks_train, paths_train = extract_features(args, extractor, train_dataloader, denoiser=denoiser)

    logger.info("Extracting test features")
    features_test, masks_test, paths_test = extract_features(args, extractor, test_dataloader, denoiser=denoiser)

    if args.load_pca_path:
        logger.info(f'Load PCA from {args.load_pca_path}')
        pca_mat = faiss.read_VectorTransform(args.load_pca_path)
    else:
        logger.info('Train PCA on train features')
        pca_mat = fit_pca(features_train, out_dim=args.pca_dim, masks=masks_train, random_sample=args.n_random_sample)
        pca_out_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_VectorTransform(pca_mat, str(pca_out_path))

    logger.info("Apply PCA on train features")
    features_train = rearrange(features_train, "n 1 t d -> (n t) d")
    features_train = pca_mat.apply_py(features_train)
    features_train = rearrange(
        features_train, "(n h w) d -> n h w d",
        n=len(train_dataloader.dataset), h=extractor.num_patches[0], w=extractor.num_patches[1]
    )

    logger.info(f"Save train features in {args.train_root}")
    save_features(paths_train, features_train, img_postfix=args.img_postfix, feat_postfix=feat_postfix)


    logger.info("Apply PCA on test features")
    features_test = rearrange(features_test, "n 1 t d -> (n t) d")
    features_test = pca_mat.apply_py(features_test)
    features_test = rearrange(
        features_test, "(n h w) d -> n h w d",
        n=len(test_dataloader.dataset), h=h, w=w
    )

    logger.info(f"Save test features in {args.test_root}")
    save_features(paths_test, features_test, img_postfix=args.img_postfix, feat_postfix=feat_postfix)


def get_default_parser():
    parser = configargparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', type=str, is_config_file=True, help='Specify a config file path')
    parser.add_argument('--train_root', type=str, help='')
    parser.add_argument('--test_root', type=str, help='')
    parser.add_argument('--exp_name', type=str, help='')
    parser.add_argument('--results_info_root', type=str, help='')
    parser.add_argument('--model_type', default='dino_vits8', type=str, help='')
    parser.add_argument('--vis_out_root', default=None, type=str, help='')
    parser.add_argument('--name_depth', default=None, type=int, help='')
    parser.add_argument('--model_path', default=None, type=str, help='')
    parser.add_argument('--load_pca_path', default=None, type=str, help='')
    parser.add_argument('--img_postfix', default='_rgb.jpg', type=str, help='')
    parser.add_argument('--mask_postfix', default='_mask.png',  type=str, help='')
    parser.add_argument('--image_size', default=224, type=int, help='')
    parser.add_argument('--img_pad', default=0, type=int, help='')
    parser.add_argument('--stride', default=4, type=int, help='')
    parser.add_argument('--facet', default='key', type=str, help='')
    parser.add_argument('--layer', default=11, type=int, help='')
    parser.add_argument('--batch_size', default=16, type=int, help='')
    parser.add_argument('--pca_dim', default=9, type=int, help='')
    parser.add_argument('--save_features_as_npy', action='store_true', help='')
    parser.add_argument('--dim_in_filename', action='store_true', help='')
    parser.add_argument('--load_mask', action='store_true', help='')
    parser.add_argument('--normalize_features', action='store_true', help='')
    parser.add_argument('--shuffle_data', action='store_true', help='')
    parser.add_argument('--n_images_vis', default=1000, type=int, help='')
    parser.add_argument('--n_random_sample', default=None, type=int, help='')
    parser.add_argument('--denoise', action="store_true", help='')
    parser.add_argument('--denoiser_ckpt_path', type=str, default=None, help='')
    parser.add_argument('--fuse_sd', action="store_true", help='')
    parser.add_argument('--featup', action="store_true", help='')
    return parser


if __name__ == '__main__':
    parser = get_default_parser()
    args, _ = parser.parse_known_args()
    main(args)
