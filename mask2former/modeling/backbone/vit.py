import torch
import torch.nn as nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


class FeedForwardNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_fc=2, drop_rate=0.0):
        super(FeedForwardNet, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_fc = num_fc

        in_dim = embed_dim
        layers = nn.ModuleList()
        for _ in range(self.num_fc - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(drop_rate),
                )
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, embed_dim))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        return self.layers(x)


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dim=768,
        kernel_size=16,
        stride=16,
        dilation=1,
        bias=True,
    ):
        super(PatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        if stride is None:
            stride = kernel_size

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class VisionTransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_head,
        hidden_dim,
        attn_drop_rate=0.0,
        drop_rate=0.0,
        num_fc=2,
        qkv_bias=True,
    ):
        super(VisionTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_head, attn_drop_rate, bias=qkv_bias
        )
        self.ffn = FeedForwardNet(embed_dim, hidden_dim, num_fc, drop_rate=drop_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.attn(self.norm1(x)) + x
        return self.ffn(self.norm2(x)) + x


@BACKBONE_REGISTRY.register()
class VisionTransformer(Backbone):
    def __init__(self, cfg, input_shape):
        super(VisionTransformer, self).__init__()

        img_size = cfg.MODEL.VIT.IMG_SIZE
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, (
                f"The size of image should have length 1 or 2, "
                f"but got {len(img_size)}"
            )

        cfg = cfg.MODEL.VIT
        self.img_size = img_size
        self.patch_size = cfg.PATCH_SIZE
        self.embed_dim = cfg.EMBED_DIM
        self.pretrained = cfg.PRETRAINED
        self.num_layers = cfg.NUM_LAYERS
        self.out_indices = cfg.OUT_INDICES
        self.with_cls_token = cfg.WITH_CLS_TOKEN
        self.use_checkpoint = cfg.USE_CHECKPOINT
        self._out_features = cfg.OUT_FEATURES
        self._frozen_stage = cfg.FROZEN_STAGE
        in_channels = 3
        drop_rate = cfg.DROP_RATE

        self.patch_embed = nn.Conv2d(
            in_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

        assert (
            img_size[0] % self.patch_size == 0 and img_size[1] % self.patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (img_size[0] // self.patch_size) * (
            img_size[1] // self.patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, self.embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.dropout = nn.Dropout(drop_rate)
        # omit drop_path

        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(
                VisionTransformerLayer(
                    embed_dim=self.embed_dim,
                    num_head=cfg.NUM_HEAD,
                    hidden_dim=int(self.embed_dim * cfg.MLP_RATIO),
                    attn_drop_rate=cfg.ATTN_DROP_RATE,
                    drop_rate=drop_rate,
                    num_fc=cfg.NUM_FC,
                    qkv_bias=cfg.QKV_BIAS,
                )
            )

        self.final_norm = None
        if cfg.FINAL_NORM:
            self.final_norm = nn.LayerNorm(self.embed_dim)

        self._freeze_stages(self._frozen_stage)

    def _freeze_stages(self, frozen_stages):
        if frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if frozen_stages >= 1:
            self.pos_embed.requires_grad = False

        if frozen_stages >= 2:
            self.cls_token.requires_grad = False

        if frozen_stages >= 3:
            self.dropout.eval()
            if self.final_norm is not None:
                self.final_norm.eval()
            for i in range(0, frozen_stages - 2):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    if "ffn" in m:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.patch_embed(x)
        h, w = x.shape[-2], x.shape[-1]
        rearrange(x, "b c h w -> b (h w) c")
        batch_size, num_patch, _ = x.shape

        cls_token = repeat(self.cls_token, "() n d -> b n d", b=batch_size)
        x = torch.cat((cls_token, x), dim=1)
        x += self.pos_embed[:, : (num_patch + 1)]
        x = self.dropout(x)

        if not self.with_cls_token:
            x = x[:, 1:]

        outs = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == self.num_layers - 1 and self.final_norm is not None:
                x = self.final_norm(x)

            if (i + 1) in self.out_indices:
                out = x[:, 1:] if self.with_cls_token else x
                rearrange(out, "b (h w) c -> b c h w", h=h)
                outs[f"res{len(outs) + 2}"] = out

        return outs

    def train(self, mode=True):
        super(VisionTransformer, self).train(mode)
        self._freeze_stages(self._frozen_stage)

    def output_shape(self):
        return {
            name: ShapeSpec(channels=self.embed_dim, stride=self.patch_size)
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32

