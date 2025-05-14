import types
import torch
from torch import nn
import timm
import timm.data
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import Block, Mlp
from functools import partial

"""
Code adapted from https://github.com/Jiawei-Yang/Denoising-ViT
Use ckpt from https://huggingface.co/jjiaweiyang/DVT/blob/main/imgnet_denoised/vit_base_patch14_dinov2.lvd142m.pth
"""

# TODO: update with latest code from the repo


class Denoiser(nn.Module):
    def __init__(
        self,
        noise_map_height: int = 37,
        noise_map_width: int = 37,
        feature_dim: int = 768,
        enable_pe: bool = True,
        denoiser_type: str = "transformer",
    ):
        super().__init__()
        self.denoiser_type = denoiser_type
        if self.denoiser_type == "transformer":
            self.denoiser = Block(
                dim=feature_dim,
                num_heads=feature_dim // 64,
                mlp_ratio=4,
                qkv_bias=True,
                qk_norm=False,
                init_values=None,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                mlp_layer=Mlp,
            )
        elif self.denoiser_type == "conv1x1":
            self.denoiser = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, 1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 2, feature_dim // 2, 1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 2, feature_dim, 1),
            )
        elif self.denoiser_type == "conv3x3":
            self.denoiser = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 2, feature_dim // 2, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature_dim // 2, feature_dim, 3, padding=1),
            )

        self.enable_pe = enable_pe
        if self.enable_pe:
            self.pos_embed = nn.Parameter(
                torch.randn(
                    1,
                    noise_map_height * noise_map_width,
                    feature_dim,
                )
                * 0.02,
                requires_grad=True,
            )
            self.large_pos_embed = None
        else:
            self.pos_embed = None
            self.large_pos_embed = None
        self.vit = None

    def forward(
        self,
        x: torch.Tensor,
        return_prefix_tokens=False,
        return_class_token=False,
        norm=True,
        return_dict=False,
        return_channel_first=False,
    ) -> torch.Tensor:
        # run backbone if backbone is there
        prefix_tokens, raw_vit_feats = None, None
        if self.vit is not None:
            with torch.no_grad():
                vit_outputs = self.vit.get_intermediate_layers(
                    x,
                    n=[self.vit.last_layer_index],
                    reshape=True,
                    return_prefix_tokens=return_prefix_tokens,
                    return_class_token=return_class_token,
                    norm=norm,
                )
                vit_outputs = (
                    vit_outputs[-1]
                    if return_prefix_tokens or return_class_token
                    else vit_outputs
                )
                raw_vit_feats = vit_outputs[0].permute(0, 2, 3, 1).detach()
                x = raw_vit_feats
                if return_prefix_tokens or return_class_token:
                    prefix_tokens = vit_outputs[1]
        B, H, W, C = x.shape
        if self.denoiser_type == "transformer":
            x = x.reshape(B, H * W, C)
            if self.enable_pe:
                pos_embed = resample_abs_pos_embed(
                    self.pos_embed, (H, W), num_prefix_tokens=0
                )
                x = x + pos_embed.repeat(B, 1, 1)
            x = self.denoiser(x)
            out_feat = x.reshape(B, H, W, C)
        else:
            out_feat = self.denoiser(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        raw_vit_feats = (
            raw_vit_feats.reshape(B, H, W, C) if raw_vit_feats is not None else None
        )
        if return_channel_first:
            out_feat = out_feat.permute(0, 3, 1, 2)
            raw_vit_feats = (
                raw_vit_feats.permute(0, 3, 1, 2) if raw_vit_feats is not None else None
            )
        if return_dict:
            return {
                "pred_denoised_feats": out_feat,
                "raw_vit_feats": raw_vit_feats,
                "prefix_tokens": prefix_tokens,
            }
        if prefix_tokens is not None:
            return out_feat, prefix_tokens
        return out_feat


class ViTWrapper(nn.Module):
    def __init__(
        self,
        model_type: str = "vit_base_patch14_dinov2.lvd142m",
        stride: int = 7,
        dynamic_img_size: bool = True,
        dynamic_img_pad: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        self.stride = stride
        self.dynamic_img_size = dynamic_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.model, self.transformation = self.create_model(model_type, **kwargs)

        p = self.model.patch_embed.patch_size
        self.patch_size = [p, p] if isinstance(p, int) else p

        if stride != self.patch_size[0]:
            stride = torch.nn.modules.utils._pair(stride)
            assert all(
                (self.patch_size[0] % s == 0 for s in stride)
            ), f"Stride {stride} should evenly divide patch size {self.patch_size[0]}."
            stride = stride[0]

        self.model.patch_embed.proj.stride = [stride, stride]
        self.override_get_intermediate()

    @property
    def n_output_dims(self) -> int:
        return self.model.pos_embed.shape[-1]

    @property
    def num_blocks(self) -> int:
        return len(self.model.blocks)

    @property
    def last_layer_index(self) -> int:
        return self.num_blocks - 1

    def create_model(
        self,
        model_type: str,
        **kwargs,
    ) -> nn.Module:
        assert model_type in [
            # DINOv1
            "vit_small_patch8_224.dino",
            "vit_small_patch16_224.dino",
            "vit_base_patch8_224.dino",
            "vit_base_patch16_224.dino",
            # DINOv2
            "vit_small_patch14_dinov2.lvd142m",
            "vit_base_patch14_dinov2.lvd142m",
            "vit_large_patch14_dinov2.lvd142m",
            "vit_giant_patch14_dinov2.lvd142m",
            # DINOv2 + register
            "vit_small_patch14_reg4_dinov2.lvd142m",
            "vit_base_patch14_reg4_dinov2.lvd142m",
            "vit_large_patch14_reg4_dinov2.lvd142m",
            "vit_giant_patch14_reg4_dinov2.lvd142m",
            # MAE
            "vit_base_patch16_224.mae",
            "vit_large_patch16_224.mae",
            "vit_huge_patch14_224.mae",
            # CLIP
            "vit_base_patch16_clip_384.laion2b_ft_in12k_in1k",
            # EVA
            "eva02_base_patch16_clip_224.merged2b",
            # DEiT-III
            "deit3_base_patch16_224.fb_in1k",
            # Auto-auged supervised ViT:
            "vit_base_patch16_384.augreg_in21k_ft_in1k",

            # commented out for simplicity. Do not use these models for now
            # it's a bit annoying to hack the intermediate layers for these models
            # however, in our informal early experiments, these models all exhibit
            # similar artifacts as the models above.
            # the artifacts in SAM are similar to the ones in MAE, while the
            # artifacts in iJEPGA are similar to the ones in DeiT.

            # SAM
            # "samvit_base_patch16.sa1b",
            # "samvit_large_patch16.sa1b",
            # "samvit_huge_patch16.sa1b",

            # ijepga
            # "vit_huge_patch14_224_ijepa.in1k",
        ]
        num_classes = 0 if "num_classes" not in kwargs else kwargs["num_classes"]
        if "num_classes" in kwargs:
            del kwargs["num_classes"]
        try:
            model = timm.create_model(
                model_type,
                pretrained=True,
                num_classes=num_classes,
                dynamic_img_size=self.dynamic_img_size,
                dynamic_img_pad=self.dynamic_img_pad,
                **kwargs,
            )
        except:
            # some model do not support dynamic_img_size and dynamic_img_pad
            model = timm.create_model(
                model_type,
                pretrained=True,
                num_classes=num_classes,
                **kwargs,
            )
        data_config = timm.data.resolve_model_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        return model, transforms

    @staticmethod
    def _hack_eva_intermediate_layers():
        def _intermediate_layers(
            self,
            x: torch.Tensor,
            n=1,
        ):
            outputs, num_blocks = [], len(self.blocks)
            take_indices = set(
                range(num_blocks - n, num_blocks) if isinstance(n, int) else n
            )

            # forward pass
            x = self.patch_embed(x)
            x, rot_pos_embed = self._pos_embed(x)
            for i, blk in enumerate(self.blocks):
                x = blk(x, rope=rot_pos_embed)
                if i in take_indices:
                    outputs.append(x)

            return outputs

        return _intermediate_layers

    @staticmethod
    def _hack_get_intermediate_layers():
        def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n=1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            return_class_token: bool = False,
            norm: bool = True,
        ):
            """Intermediate layer accessor (NOTE: This is a WIP experiment).
            Inspired by DINO / DINOv2 interface
            """
            # take last n blocks if n is an int, if in is a sequence, select by matching indices
            outputs = self._intermediate_layers(x, n)
            if norm:
                outputs = [self.norm(out) for out in outputs]
            if return_class_token:
                # prefix_tokens = [out[:, 0:1] for out in outputs]
                prefix_tokens = [out[:, 0] for out in outputs]
            else:
                prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
            outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

            if reshape:
                B, C, H, W = x.shape
                grid_size = (
                    (H - self.patch_embed.patch_size[0])
                    // self.patch_embed.proj.stride[0]
                    + 1,
                    (W - self.patch_embed.patch_size[1])
                    // self.patch_embed.proj.stride[1]
                    + 1,
                )
                outputs = [
                    out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    for out in outputs
                ]

            if return_prefix_tokens or return_class_token:
                return tuple(zip(outputs, prefix_tokens))
            return tuple(outputs)

        return get_intermediate_layers

    def override_get_intermediate(self) -> None:
        if "eva" in self.model_type:
            self.model._intermediate_layers = types.MethodType(
                ViTWrapper._hack_eva_intermediate_layers(),
                self.model,
            )
        self.model.get_intermediate_layers = types.MethodType(
            ViTWrapper._hack_get_intermediate_layers(),
            self.model,
        )

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n=1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        return self.model.get_intermediate_layers(
            x,
            n=n,
            reshape=reshape,
            return_prefix_tokens=return_prefix_tokens,
            return_class_token=return_class_token,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
