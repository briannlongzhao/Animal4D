import logging
import numpy as np
import mmcv
from mmcv.runner import load_checkpoint
from mmpose.apis import inference_top_down_pose_model, vis_pose_result
from mmpose.datasets import DatasetInfo
from mmpose.models import build_posenet


class ViTPose:
    def __init__(
        self, cfg_path, ckpt_path, device, return_heatmap=False, output_layer_names=None, radius=8,
        thickness=4, kpt_thr=0.3, return_pose_image=False
    ):
        self.model = self.init_pose_model(cfg_path, ckpt_path, device=device)
        self.dataset = self.model.cfg.data['test']['type']
        dataset_info = self.model.cfg.data['test'].get('dataset_info', None)
        self.dataset_info = DatasetInfo(dataset_info)
        self.return_heatmap = return_heatmap
        self.output_layer_names = output_layer_names
        self.radius = radius
        self.kpt_thr = kpt_thr
        self.thickness = thickness
        self.return_pose_image = return_pose_image

    @staticmethod
    def init_pose_model(config, checkpoint, device):
        """Rewrite mmpose.apis.init_pose_model for silent logging"""
        if isinstance(config, str):
            config = mmcv.Config.fromfile(config)
        elif not isinstance(config, mmcv.Config):
            raise TypeError('config must be a filename or Config object, '
                            f'but got {type(config)}')
        config.model.pretrained = None
        model = build_posenet(config.model)
        if checkpoint is not None:
            # load model checkpoint
            logger = logging.getLogger()
            logger.addHandler(logging.NullHandler())
            logger.propagate = False
            load_checkpoint(model, checkpoint, map_location='cpu', logger=logger)
        # save the config in the model for convenience
        model.cfg = config
        model.to(device)
        model.eval()
        return model

    def __call__(self, image):
        h, w = image.shape[:2]
        person_results = [{'bbox': np.array([0, 0, w, h])}]
        pose_results, returned_outputs = inference_top_down_pose_model(
            self.model,
            image,
            person_results,
            format='xyxy',
            dataset=self.dataset,
            dataset_info=self.dataset_info,
            return_heatmap=self.return_heatmap,
            outputs=self.output_layer_names
        )
        if self.return_pose_image:
            vis_img = vis_pose_result(
                self.model,
                image,
                pose_results,
                radius=self.radius,
                thickness=self.thickness,
                dataset=self.dataset,
                dataset_info=self.dataset_info,
                kpt_score_thr=self.kpt_thr,
                show=False
            )
            return pose_results, vis_img
        else:
            return pose_results
