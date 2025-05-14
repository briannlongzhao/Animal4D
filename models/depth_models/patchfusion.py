import sys
import copy
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.transforms import Compose
from torchvision.transforms.functional import pil_to_tensor

sys.path.append("externals/PatchFusion")
import externals.PatchFusion.infer_user as PF
from externals.PatchFusion.zoedepth.utils.config import get_config_user
from externals.PatchFusion.zoedepth.models.builder import build_model
from externals.PatchFusion.zoedepth.models.base_models.midas import Resize


class PatchFusion:
    def __init__(self, device):
        model_cfg_path = "externals/PatchFusion/zoedepth/models/zoedepth_custom/configs/config_zoedepth_patchfusion.json"
        self.ckp_path = "externals/PatchFusion/nfs/patchfusion_u4k.pt"
        self.model = "zoedepth_custom"
        self.model_cfg_path = model_cfg_path
        self.mode = "p16"
        self.boundary = 0
        self.blr_mask = True
        self.device = device
        self.overwrite_kwargs = {}
        self.overwrite_kwargs['model_cfg_path'] = model_cfg_path
        self.overwrite_kwargs["model"] = self.model
        self.transform = Compose([
            Resize(512, 384, keep_aspect_ratio=False, ensure_multiple_of=32, resize_method="minimal")
        ])
        config = get_config_user(self.model, **self.overwrite_kwargs)
        config["pretrained_resource"] = ''
        self.model = build_model(config)
        self.model.to(self.device)
        self.model = PF.load_ckpt(self.model, self.ckp_path)
        self.model.eval()
        self.model = self.model.to(self.device)

    def __call__(self, images):
        """
        Input list of RGB PIL images [(w h c)]
        Return average depth maps tensor (b h w)
        """
        # TODO: support for batched input (current bsz=1)
        h, w = images[0].height, images[0].width
        crop_size = (int(h // 4), int(w // 4))
        img = np.array(images) / 255.0  # (b h w c)
        img = F.interpolate(  # (b c w h)
            torch.from_numpy(img).permute(0, 3, 1, 2), (h, w), mode='bicubic', align_corners=True
        ).detach()
        img_lr = self.transform(img)
        img = img.to(self.device)
        avg_depth_map = PF.regular_tile(
            self.model, img, offset_x=0, offset_y=0, img_lr=img_lr, crop_size=crop_size, img_resolution=(h, w),
            transform=self.transform
        )

        if self.mode == 'p16':
            pass
        elif self.mode == 'p49':
            PF.regular_tile(self.model, img, offset_x=crop_size[1] // 2, offset_y=0, img_lr=img_lr,
                            iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True,
                            avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size=crop_size,
                            img_resolution=(h, w), transform=self.transform)
            PF.regular_tile(self.model, img, offset_x=0, offset_y=crop_size[0] // 2, img_lr=img_lr,
                            iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True,
                            avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size=crop_size,
                            img_resolution=(h, w), transform=self.transform)
            PF.regular_tile(self.model, img, offset_x=crop_size[1] // 2, offset_y=crop_size[0] // 2,
                            img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True,
                            avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size=crop_size,
                            img_resolution=(h, w), transform=self.transform)

        elif self.mode[0] == 'r':
            PF.regular_tile(self.model, img, offset_x=crop_size[1] // 2, offset_y=0, img_lr=img_lr,
                            iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True,
                            avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size=crop_size,
                            img_resolution=(h, w), transform=self.transform)
            PF.regular_tile(self.model, img, offset_x=0, offset_y=crop_size[0] // 2, img_lr=img_lr,
                            iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True,
                            avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size=crop_size,
                            img_resolution=(h, w), transform=self.transform)
            PF.regular_tile(self.model, img, offset_x=crop_size[1] // 2, offset_y=crop_size[0] // 2,
                            img_lr=img_lr, iter_pred=avg_depth_map.average_map, boundary=self.boundary, update=True,
                            avg_depth_map=avg_depth_map, blr_mask=self.blr_mask, crop_size=crop_size,
                            img_resolution=(h, w), transform=self.transform)

            for i in tqdm(range(int(self.mode[1:]))):
                PF.random_tile(self.model, img, img_lr=img_lr, iter_pred=avg_depth_map.average_map,
                               boundary=self.boundary, update=True, avg_depth_map=avg_depth_map, blr_mask=self.blr_mask,
                               crop_size=crop_size, img_resolution=(h, w), transform=self.transform)

        color_depth_map = copy.deepcopy(avg_depth_map.average_map)
        color_depth_map = PF.colorize_infer(color_depth_map.detach().cpu().numpy())

        return avg_depth_map.average_map.unsqueeze(0)
