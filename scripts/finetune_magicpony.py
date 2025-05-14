import os
from pathlib import Path
from shutil import rmtree, copy2
import cv2
import sys
import torch
import hydra
import traceback
from glob import glob
from tqdm import tqdm
from configargparse import ArgumentParser
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from random import shuffle
from subprocess import run

submodule_dir = os.path.abspath("externals/Animals")
sys.path.append(submodule_dir)
from models.animal_models import ImageDataset, MagicponyFinetune, finetune
from externals.Animals.model.utils.wandb_writer import WandbWriter
from externals.Animals.model.utils.misc import setup_runtime


"""Finetune pretrained MagicPony model with keypoint supervision"""
"""Not tested yet"""



def update_cfg(cfg: DictConfig):
    with open_dict(cfg):

        cfg.model.cfg_loss.keypoint_projection_loss_weight = 50
        cfg.model.cfg_predictor_instance.cfg_articulation.use_fauna_constraints = False

        # cfg.log_image_freq = 1

        cfg.finetune_iter = 30000
        cfg.archive_code = False
        cfg.dataset.load_keypoint = True
        cfg.checkpoint_path = os.path.join(submodule_dir, "results/magicpony/horse_v2_backup/checkpoint140000.pth")

        cfg.model.cfg_loss.mask_inv_dt_loss_weight = 500
        cfg.model.cfg_loss.logit_loss_dino_feat_im_loss_multiplier = 200
        cfg.model.cfg_predictor_instance.cfg_articulation.enable_refine = False
        cfg.model.cfg_predictor_instance.cfg_articulation.legs_to_body_joint_indices = [3, 6, 6, 3]
        cfg.model.cfg_predictor_instance.cfg_pose.rand_campos = False
        cfg.dataset.num_workers = 1
        cfg.dataset.in_image_size = 512
    return cfg


@hydra.main(config_path=os.path.join(submodule_dir, "config"), config_name="train_magicpony_horse")
def main(cfg: DictConfig):
    cfg = update_cfg(cfg)
    if cfg.logger_type == "wandb":
        logger = WandbWriter(project="FinetuneMagicpony", config=cfg, local_dir=cfg.dataset.local_dir)
    elif cfg.logger_type == "tensorboard":
        raise NotImplementedError("Tensorboard logger not implemented, use logger_type=wandb or unset logger_type")
    else:
        logger = None
    device = setup_runtime(cfg)
    accelerator = Accelerator()
    model = MagicponyFinetune(cfg.model)
    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    cp = torch.load(cfg.checkpoint_path, map_location="cpu")
    epoch, total_iter = cp["epoch"], cp["total_iter"]

    model.reset_optimizers()
    model.set_train()
    model.load_model_state(cp)
    model.set_train_post_load()
    model.to(device)

    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            setattr(model, name, accelerator.prepare_model(value))
        if isinstance(value, torch.optim.Optimizer):
            setattr(model, name, accelerator.prepare_optimizer(value))
        if isinstance(value, torch.optim.lr_scheduler._LRScheduler):
            setattr(model, name, accelerator.prepare_scheduler(value))
    model.accelerator = accelerator

    # Train data
    train_dataset = ImageDataset(
        cfg.dataset.train_data_dir,
        in_image_size=cfg.dataset.in_image_size,
        out_image_size=cfg.dataset.out_image_size,
        # load_dino_feature=cfg.dataset.load_dino_feature,
        load_keypoint=cfg.dataset.load_keypoint,
    )
    test_dataset = ImageDataset(
        cfg.dataset.test_data_dir,
        in_image_size=cfg.dataset.in_image_size,
        out_image_size=cfg.dataset.out_image_size,
        # load_dino_feature=cfg.dataset.load_dino_feature,
        load_keypoint=cfg.dataset.load_keypoint,
    )
    train_dataloader = DataLoader(
        train_dataset, num_workers=cfg.dataset.num_workers, batch_size=cfg.dataset.batch_size,
        pin_memory=True, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, num_workers=cfg.dataset.num_workers, batch_size=cfg.dataset.batch_size,
        pin_memory=True, shuffle=False,
    )
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    model = finetune(
        model, train_dataloader, test_dataloader, logger,
        total_iter, epoch, cfg.finetune_iter, cfg.log_image_freq, cfg.log_loss_freq, device
    )



def validate_tensor_to_device(x, device=None):
    if type(x) is not torch.Tensor:
        return x
    elif torch.any(torch.isnan(x)):
        return None
    elif device is None:
        return x
    else:
        return x.to(device)

def validate_all_to_device(batch, device=None):
    return tuple(validate_tensor_to_device(x, device) for x in batch)


if __name__ == "__main__":
    main()
