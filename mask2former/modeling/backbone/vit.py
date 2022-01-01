import logging

from numpy import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from ...utils.vit_loader_helper import load_weights_from_npz


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
        out = self.norm1(x)
        out = self.attn(out, out, out)[0] + x
        return self.ffn(self.norm2(out)) + out


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
        self.weights = cfg.WEIGHTS
        self.interpolate_mode = cfg.POS_EMBED_INTERPOLATE_MODE
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

        self.logger = logging.getLogger(__name__)

        self._freeze_stages(self._frozen_stage)
        self.init_weights(self.weights)

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
        if pretrained is not None:
            state_dict = pretrained
            if isinstance(pretrained, str) and \
                pretrained.endswith(".npz"):
                self.logger.warning(f"ViT load weights from {pretrained}")
                state_dict = load_weights_from_npz(self, pretrained)
            if 'pos_embed' in state_dict.keys() and \
                self.pos_embed.shape != state_dict['pos_embed'].shape:
                self.logger.warning(
                    f'Resize the pos_embed shape from '
                    f'{state_dict["pos_embed"].shape} to '
                    f'{self.pos_embed.shape}'
                )
                h, w = self.img_size
                pos_size = int(sqrt(state_dict['pos_embed'].shape[1] - 1))
                state_dict['pos_embed'] = self._resize_pos_embed(
                    state_dict['pos_embed'],
                    (h // self.patch_size, w // self.patch_size),
                    (pos_size, pos_size), self.interpolate_mode
                )
            self.load_state_dict(state_dict, False)

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

    def _resize_pos_embed(self, pos_embed, input_shape, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0].unsqueeze(1)
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = rearrange(pos_embed_weight, 
            "b (h w) c -> b c h w", h = pos_h)
        pos_embed_weight = F.interpolate(
            pos_embed_weight, 
            size=input_shape, 
            mode=mode,
            align_corners=False
        )
        pos_embed_weight = rearrange(pos_embed_weight, "b c h w -> b (h w) c")
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def _pos_embed(self, patched_img, hw_shape, pos_embed):
        """Position embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        if patched_img.shape[1] != pos_embed.shape[1]:
            pos_embed = self._resize_pos_embed(
                pos_embed,
                hw_shape,
                (self.img_size[0] // self.patch_size, 
                    self.img_size[1] // self.patch_size),
                self.interpolate_mode
            )
        return self.dropout(patched_img + pos_embed)

    def _embed(self, img):
        '''Create Embeded images.

        Transfer image tensor into transformer input, 
            including patch embeding and position embeding.
        Args:
            img (torch.Tensor): The image, it should be shape of [B, C, H, W].
        Return:
            torch.Tensor: The embeded image feature, 
                which is sended into transformer, 
                it would be shape of [B, L, C].
            tuple: the 2D shape of the embeded image 
        '''
        x = self.patch_embed(img)
        batch_size, _, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")

        cls_token = repeat(self.cls_token, "() n d -> b n d", b=batch_size)
        x = torch.cat((cls_token, x), dim=1)
        x = self._pos_embed(x, (h, w), self.pos_embed)

        if not self.with_cls_token:
            x = x[:, 1:]

        return x, (h, w)

    def forward(self, x):
        x, (h, w) = self._embed(x)

        outs = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if (i + 1) in self.out_indices:
                out = x[:, 1:] if self.with_cls_token else x
                out = self.final_norm(out)
                out = rearrange(out, "b (h w) c -> b c h w", h=h)
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

