import torch
from pathlib import Path
from configargparse import ArgumentParser
from models.feature_extractor import FeatureExtractor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, is_config_file=True, help="Path to yaml config file")
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory containing train and test")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for dataset splitting")
    parser.add_argument(
        "--pca_mode", type=str, required=True, choices=["fit", "apply"],
        help="Run only fit pca or load and apply pca to image feature"
    )
    parser.add_argument(
        "--model_type", type=str, default="dinov2_vits14",
        choices=["dinov2_vitb14", "dinov2_vits14"],
        help="Feature extractor model type, 768 dim for dinov2_vitb14, 384 dim for dinov2_vits14"
    )
    parser.add_argument(
        "--stride", type=int, default=7, help="Stride of the feature extractor"
    )
    parser.add_argument(
        "--image_size", type=int, default=560, help="Image size of the feature extractor"
    )
    parser.add_argument(
        "--image_pad", type=int, default=0, help="Padding of the image"
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Denoise the image feature"
    )
    parser.add_argument(
        "--pca_dim", type=int, default=16, help="PCA dimension"
    )
    parser.add_argument(
        "--pca_path", type=str, default=None, help="Path to save/load PCA model"
    )
    parser.add_argument(
        "--feature_batch_size", type=int, default=2, help="Batch size to extract image feature"
    )
    parser.add_argument(
        "--pca_fit_samples", type=int, default=2500, help="Number of random samples to fit PCA"
    )
    parser.add_argument(
        "--image_suffix", type=str, default="rgb.png", help="Suffix of the image file"
    )
    parser.add_argument(
        "--mask_suffix", type=str, default="mask.png", help="Suffix of the mask file"
    )
    parser.add_argument(
        "--feature_suffix", type=str, default="feature.png", help="Suffix of the feature file"
    )
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.pca_path is None:
        args.pca_path = Path(args.dataset_dir) / f"pca_{args.model_type}_{args.pca_dim}.bin"
    feature_extractor = FeatureExtractor(
        dataset_dir=args.dataset_dir,
        pca_mode=args.pca_mode,
        model_type=args.model_type,
        stride=args.stride,
        image_size=args.image_size,
        image_pad=args.image_pad,
        denoise=args.denoise,
        pca_dim=args.pca_dim,
        feature_batch_size=args.feature_batch_size,
        pca_fit_samples=args.pca_fit_samples,
        image_suffix=args.image_suffix,
        mask_suffix=args.mask_suffix,
        seed=args.seed,
        device=device,
        pca_path=args.pca_path,
        feature_suffix=args.feature_suffix,
    )
    feature_extractor.run()
