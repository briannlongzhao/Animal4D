import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


class DepthAnything:
    def __init__(self, device):
        model_url = "LiheYoung/depth-anything-small-hf"
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_url)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_url).to(self.device)

    def __call__(self, images):
        """
        Input single or list of RGB PIL images [(w h c)]
        Return average depth maps tensor (b h w)
        """
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=images[0].size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        output = prediction.squeeze(1)
        return output
