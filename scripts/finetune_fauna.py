import os
import sys
import torch
import hydra
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from accelerate import Accelerator
from datetime import datetime

submodule_dir = os.path.abspath("externals/Animals")
sys.path.append(submodule_dir)

from models.animal_models.models import FaunaFinetune
from models.animal_models.datasets import FaunaImageDataset
from models.animal_models.utils import finetune, save_checkpoint
from externals.Animals.model.utils.wandb_writer import WandbWriter
from externals.Animals.model.utils.misc import setup_runtime



def update_cfg(cfg: DictConfig):
    with open_dict(cfg):
        cfg.dataset.train_data_dir = "data/data_3.0.0/train/"
        cfg.dataset.test_data_dir = "data/data_3.0.0/train/"
        cfg.checkpoint_path = os.path.join(submodule_dir, "results/fauna/pretrained_fauna/pretrained_fauna.pth")
        cfg.checkpoint_dir = os.path.join(submodule_dir, "results/fauna/fauna_finetune")

        cfg.model.cfg_loss.keypoint_projection_loss_weight = 25
        cfg.finetune_iter = 30000
        cfg.archive_code = False
        cfg.dataset.load_keypoint = True
        cfg.dataset.load_dino_feature = True
        cfg.model.cfg_predictor_instance.cfg_articulation.enable_refine = False
        cfg.model.cfg_predictor_instance.cfg_pose.rand_campos = False
        cfg.dataset.local_dir = "/scr-ssd/briannlz"
        cfg.dataset.in_image_size = 512
        cfg.dataset.num_workers = 1
        cfg.logger_type = "wandb"
        cfg.use_logger = True
        cfg.dataset.batch_size = 6
        cfg.model.cfg_predictor_instance.cfg_pose.rand_campos = False
        # cfg.log_image_freq = 1
    return cfg


@hydra.main(config_path=os.path.join(submodule_dir, "config"), config_name="train_fauna")
def main(cfg: DictConfig):
    cfg = update_cfg(cfg)
    if cfg.use_logger:
        if cfg.logger_type == "wandb":
            logger = WandbWriter(project="FinetuneFauna", config=cfg, local_dir=cfg.dataset.local_dir)
        elif cfg.logger_type == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter
            logger = SummaryWriter(
                os.path.join(cfg.dataset.local_dir, 'tensorboard_logs', datetime.now().strftime("%Y%m%d-%H%M%S")),
                flush_secs=10
            )
        else:
            raise ValueError(f"Unknown logger type: {cfg.logger_type}")
    else:
        logger = None
    device = setup_runtime(cfg)
    accelerator = Accelerator()
    model = FaunaFinetune(cfg.model)
    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    cp = torch.load(cfg.checkpoint_path, map_location="cpu")
    epoch, total_iter = cp["epoch"], cp["total_iter"]

    model.reset_optimizers()
    model.set_train()
    model.load_model_state(cp)
    model.set_train_post_load()
    model.to(device)

    print("Preparing")
    for name, value in vars(model).items():
        if isinstance(value, torch.nn.Module):
            setattr(model, name, accelerator.prepare_model(value))
        if isinstance(value, torch.optim.Optimizer):
            setattr(model, name, accelerator.prepare_optimizer(value))
        if isinstance(value, torch.optim.lr_scheduler._LRScheduler):
            setattr(model, name, accelerator.prepare_scheduler(value))
    model.accelerator = accelerator

    print("Loading data")
    train_dataset = FaunaImageDataset(
        cfg.dataset.train_data_dir,
        in_image_size=cfg.dataset.in_image_size,
        out_image_size=cfg.dataset.out_image_size,
        load_keypoint=cfg.dataset.load_keypoint,
        load_dino_feature=cfg.dataset.load_dino_feature,
        batch_size=cfg.dataset.batch_size,
        load_in_chunks=True,
        shuffle=True,
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,
    )
    test_dataset = FaunaImageDataset(
        cfg.dataset.test_data_dir,
        in_image_size=cfg.dataset.in_image_size,
        out_image_size=cfg.dataset.out_image_size,
        load_keypoint=cfg.dataset.load_keypoint,
        load_dino_feature=cfg.dataset.load_dino_feature,
        batch_size=cfg.dataset.batch_size,
        load_in_chunks=True,
        shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset,
        num_workers=cfg.dataset.num_workers,
        pin_memory=True,
        shuffle=False,
        batch_size=cfg.dataset.batch_size,
    )
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    model = finetune(
        model, train_dataloader, test_dataloader, logger,
        total_iter, epoch, cfg.finetune_iter, cfg.log_image_freq, cfg.log_loss_freq, device
    )

    save_checkpoint(model=model, checkpoint_dir=cfg.checkpoint_dir, epoch=epoch, total_iter=total_iter)



if __name__ == "__main__":
    main()



# python run.py --config-name test_fauna +checkpoint_path=results/fauna/pretrained_fauna/pretrained_fauna.pth test_data_dir=data/fauna/data_3.0.0/test/wolf/-HuVqPKgpII_329_001/ test_result_dir=results/temp
