import sys
import torch
import numpy as np
import torch.nn.functional as F
sys.path.insert(0, "externals/SEA_RAFT/core")
from externals.SEA_RAFT.core.raft import RAFT
from externals.SEA_RAFT.config.parser import json_to_args
from externals.SEA_RAFT.core.utils.flow_viz import flow_to_image


class SEA_RAFT:
    def __init__(self, device):
        config_path = "externals/SEA_RAFT/config/eval/spring-M.json"
        model_path = "MemorySlices/Tartan-C-T-TSKH-spring540x960-M"
        args = json_to_args(config_path)
        args.cfg = config_path
        args.url = model_path
        self.device = device
        self.scale = args.scale
        self.iters = args.iters
        self.model = RAFT.from_pretrained(model_path, args=args).to(device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image1, image2):
        image1 = image1.to(self.device)
        image2 = image2.to(self.device)
        image1 = F.interpolate(image1, scale_factor=2 ** self.scale, mode='bilinear', align_corners=False)
        image2 = F.interpolate(image2, scale_factor=2 ** self.scale, mode='bilinear', align_corners=False)
        output = self.model(image1, image2, iters=self.iters, test_mode=True)
        flows, info = output['flow'][-1], output['info'][-1]
        flows = F.interpolate(
            flows, scale_factor=0.5 ** self.scale, mode='bilinear', align_corners=False
        ) * (0.5 ** self.scale)
        # info_down = F.interpolate(info, scale_factor=0.5 ** self.scale, mode='area')
        flow_images = np.stack([  # TODO: maybe batchify flow_to_image
            flow_to_image(flow.permute(1, 2, 0).cpu().numpy(), convert_to_bgr=True) for flow in flows
        ])
        return flows, flow_images


