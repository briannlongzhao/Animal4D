import os
import sys
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from random import shuffle, choice
from pathlib import Path
from diffusers import (
    StableDiffusion3ControlNetPipeline, StableDiffusionControlNetPipeline, ControlNetModel,
    StableDiffusionXLControlNetPipeline, StableDiffusionXLPipeline, FluxControlPipeline,
    FluxTransformer2DModel, FluxControlImg2ImgPipeline
)
from diffusers.utils import load_image
from diffusers.models import SD3ControlNetModel
from diffusers.image_processor import VaeImageProcessor
from transformers import SiglipVisionModel, SiglipImageProcessor
from image_gen_aux import DepthPreprocessor



"""
Given a data directory containing MagicPony rendered depth/canny, articulation, and mask,
generate real rgb images using depth/canny as input to ControlNet, optionally with a style image condition.
"""


data_dir = Path("data/magicpony_sdxl_canny/val/")
diffusion_model = "sdxl"  # sd1.5, sd3.5, sdxl, flux
control_image_suffix = "canny.png"
# style_image_paths = list(Path("data/data_2.0.0.2/").glob("**/*_rgb.png"))
style_image_paths = list(Path("data/magicpony_horse_v2/").glob("**/*_rgb.*"))
category = "horse"
prompts = [
    "A photo of a {}",
    "A {} ",
    "A natural photograph of a {}",
    "A {} with natural background",
    "Realistic photograph of a {}",
    "Genuine camera shot of a {}",
    "Photoreal close-up portrait of a {}",
    "Natural photograph of a {}",
    "Close shot of a {}",
    "True-to-life photograph of a {}",
    "Simple camera shot of a {}",
    "Lifelike photo of a {}"
]
# colors = ["dark red", "brick color", "black", "chestnut color", "bay color", "roan color", "champagne color", "silver color", "white", "brown", "yellow", "gray", ""]
a_prompt = "natural background, nature scene, natural color, highres, visible legs"
n_prompt = "monochrome, lowres, unreal, bad anatomy, worst quality, low quality"

replace_dict = {
    "depth_pred_normalized.png": "depth.png",
    "mask_pred.png": "mask.png",
    "arti_params.txt": "articulation.txt",
    "edge.png": "canny.png",
}


def load_pipe(method):
    if method == "sd1.5":
        assert "depth" in control_image_suffix, "Control image must be depth"
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16,
            safety_checker=None
        )
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
    elif method == "sd3.5":
        assert "canny" in control_image_suffix, "Control image must be canny"
        class SD3CannyImageProcessor(VaeImageProcessor):
            def __init__(self):
                super().__init__(do_normalize=False)

            def preprocess(self, image, **kwargs):
                image = super().preprocess(image, **kwargs)
                image = image * 255 * 0.5 + 0.5
                return image

            def postprocess(self, image, do_denormalize=True, **kwargs):
                do_denormalize = [True] * image.shape[0]
                image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
                return image

        model_id = "stabilityai/stable-diffusion-3.5-large"
        image_encoder_id = "google/siglip-so400m-patch14-384"
        ip_adapter_id = "InstantX/SD3.5-Large-IP-Adapter"
        controlnet_id = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
        controlnet = SD3ControlNetModel.from_pretrained(
            controlnet_id, torch_dtype=torch.float16
        )
        feature_extractor = SiglipImageProcessor.from_pretrained(
            image_encoder_id, torch_dtype=torch.float16
        )
        image_encoder = SiglipVisionModel.from_pretrained(
            image_encoder_id, torch_dtype=torch.float16
        )
        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            controlnet=controlnet
        )
        pipe.image_processor = SD3CannyImageProcessor()
        pipe.load_ip_adapter(ip_adapter_id, revision="f1f54ca369ae759f9278ae9c87d46def9f133c78")
        pipe.set_ip_adapter_scale(0.7)
        pipe._exclude_from_cpu_offload.append("image_encoder")
        pipe.enable_sequential_cpu_offload()
    elif method == "sdxl":
        # Use IP-Adapter original repo
        sys.path.append("externals/IP_Adapter")
        from ip_adapter import IPAdapterXL
        controlnet_model = "diffusers/controlnet-depth-sdxl-1.0" if "depth" in control_image_suffix else "diffusers/controlnet-canny-sdxl-1.0"
        controlnet = ControlNetModel.from_pretrained(
            controlnet_model, variant="fp16", use_safetensors=True, torch_dtype=torch.float16
        ).to("cuda")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet, use_safetensors=True, torch_dtype=torch.float16, add_watermarker=False,
        ).to("cuda")
        pipe = IPAdapterXL(
            pipe,
            "externals/IP_Adapter/models/image_encoder",
            "externals/IP_Adapter/sdxl_models/ip-adapter_sdxl_vit-h.bin",
            "cuda"
        )
    elif method == "flux":
        pipe = FluxControlImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.float16)
    else:
        raise NotImplementedError
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except:
        pass
    # pipe.enable_model_cpu_offload()
    try:
        pipe.to("cuda", torch.float16)
    except:
        pass
    return pipe


def process_diffusers(pipe, method, control_image, prompt, a_prompt, n_prompt, style_image_path=None, h=256, w=256):
    if a_prompt:
        prompt = prompt + ", " + a_prompt
    control_image = Image.fromarray(control_image).convert("RGB")
    style_image = None
    if style_image_path is not None:
        print("Using style image", style_image_path)
        style_image = Image.open(style_image_path).convert("RGB")
    control_image = control_image.resize((1024, 1024))
    if style_image:
        style_image = style_image.resize((1024, 1024))
    if method == "sd1.5":
        ref_style_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png")
        ref_depth_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png")
        image = pipe(
            prompt=prompt, negative_prompt=n_prompt,
            image=control_image,  # PIL (0-255, c=3)
            ip_adapter_image=style_image,  # PIL (0-255, c=3)
            num_inference_steps=50,
        ).images[0]
    elif method == "sd3.5":
        image = pipe(
            prompt=prompt, negative_prompt=n_prompt,
            width=1024, height=1024,
            num_images_per_prompt=1,
            ip_adapter_image=style_image,
            control_image=control_image,
            controlnet_conditioning_scale=1.2,
            guidance_scale=3.5,
            num_inference_steps=50,
        ).images[0]
    elif method == "sdxl":
        ref_style_image = Image.open("externals/IP_Adapter/assets/images/statue.png")
        ref_control_image = Image.open("externals/IP_Adapter/assets/structure_controls/depth.png")
        image = pipe.generate(
            pil_image=style_image, image=control_image, prompt=prompt, negative_prompt=n_prompt,
            controlnet_conditioning_scale=0.8, num_samples=1, num_inference_steps=50
        )[0]
    elif method == "flux":
        image = pipe(
            image=style_image, prompt=prompt, #negative_prompt=n_prompt,
            control_image=control_image,  # PIL (0-255, c=3)
            num_inference_steps=30, guidance_scale=7.0, #controlnet_conditioning_scale=5.0,
        ).images[0]
    else:
        raise NotImplementedError
    image = image.resize((h, w))
    return np.array(image)


if __name__ == "__main__":
    all_files = list(data_dir.rglob("*"))
    for f in tqdm(all_files, desc="Renaming files"):
        filename = str(f)
        for old_suffix, new_suffix in replace_dict.items():
            if filename.endswith(old_suffix):
                new_filename = filename.replace(old_suffix, new_suffix)
                os.rename(filename, new_filename)
    all_files = list(data_dir.rglob("*"))
    shuffle(all_files)
    pipe = load_pipe(diffusion_model)
    for f in tqdm(all_files, desc="Generating images"):
        if not f.name.endswith(control_image_suffix):
            continue
        # box_path = str(f).replace(control_image_suffix, "box.txt")
        # with open(box_path, 'w') as box_f:  # Write dummy box.txt
        #     box_f.write("0004630_00 957.00 168.00 506.00 506.00 1920.00 1080.00 10.38")
        rgb_path = str(f).replace(control_image_suffix, f"rgb.png")
        if os.path.exists(rgb_path):
            continue
        prompt = choice(prompts).format(f"{category}")
        print(prompt)
        image = cv2.imread(str(f))
        result = process_diffusers(
            pipe=pipe, method=diffusion_model, control_image=image, prompt=prompt, a_prompt=a_prompt, n_prompt=n_prompt,
            style_image_path=choice(style_image_paths)
        )
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        cv2.imwrite(rgb_path, result)
