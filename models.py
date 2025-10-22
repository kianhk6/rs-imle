import torch
from torch import nn
from torch.nn import functional as F

from mapping_network import MappingNetowrk, AdaptiveInstanceNorm, NoiseInjection
from helpers.imle_helpers import get_1x1, get_3x3, draw_gaussian_diag_samples, gaussian_analytical_kl
from collections import defaultdict
import numpy as np
import itertools
from dit import DiT_S_2
from unet.unet import UNetModelWrapper

class Block(nn.Module):
    def __init__(self, in_width, middle_width, out_width, down_rate=None, residual=False, use_3x3=True, zero_last=False):
        super().__init__()
        self.down_rate = down_rate
        self.residual = residual
        self.c1 = get_1x1(in_width, middle_width)
        self.c2 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c3 = get_3x3(middle_width, middle_width) if use_3x3 else get_1x1(middle_width, middle_width)
        self.c4 = get_1x1(middle_width, out_width, zero_weights=zero_last)

    def forward(self, x):
        xhat = self.c1(F.gelu(x))
        xhat = self.c2(F.gelu(xhat))
        xhat = self.c3(F.gelu(xhat))
        xhat = self.c4(F.gelu(xhat))
        out = x + xhat if self.residual else xhat
        if self.down_rate is not None:
            out = F.avg_pool2d(out, kernel_size=self.down_rate, stride=self.down_rate)
        return out


def parse_layer_string(s):
    layers = []
    for ss in s.split(','):
        if 'x' in ss:
            res, num = ss.split('x')
            count = int(num)
            layers += [(int(res), None) for _ in range(count)]
        elif 'm' in ss:
            res, mixin = [int(a) for a in ss.split('m')]
            layers.append((res, mixin))
        elif 'd' in ss:
            res, down_rate = [int(a) for a in ss.split('d')]
            layers.append((res, down_rate))
        else:
            res = int(ss)
            layers.append((res, None))
    return layers


def pad_channels(t, width):
    d1, d2, d3, d4 = t.shape
    empty = torch.zeros(d1, width, d3, d4, device=t.device)
    empty[:, :d2, :, :] = t
    return empty


def get_width_settings(width, s):
    mapping = defaultdict(lambda: width)
    if s:
        s = s.split(',')
        for ss in s:
            k, v = ss.split(':')
            mapping[int(k)] = int(v)
    return mapping


class DecBlock(nn.Module):
    def __init__(self, H, res, mixin, n_blocks):
        super().__init__()
        self.base = res
        self.mixin = mixin
        self.H = H
        self.widths = get_width_settings(H.width, H.custom_width_str)
        width = self.widths[res]
        if res <= H.max_hierarchy:
            self.noise = NoiseInjection(width)
        self.adaIN = AdaptiveInstanceNorm(width, H.latent_dim)
        use_3x3 = res > 2
        cond_width = int(width * H.bottleneck_multiple)
        self.resnet = Block(width, cond_width, width, residual=True, use_3x3=use_3x3)
        self.resnet.c4.weight.data *= np.sqrt(1 / n_blocks)

    def forward(self, x, w, spatial_noise):
        if self.mixin is not None:
            x = F.interpolate(x, scale_factor=self.base // self.mixin)
        if self.base <= self.H.max_hierarchy:
            x = self.noise(x, spatial_noise)
        x = self.adaIN(x, w)
        x = self.resnet(x)
        return x


class Decoder(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.H = H
        self.mapping_network = MappingNetowrk(code_dim=H.latent_dim, n_mlp=H.n_mpl)
        resos = set()
        cond_width = int(H.width * H.bottleneck_multiple)
        dec_blocks = []
        self.widths = get_width_settings(H.width, H.custom_width_str)
        blocks = parse_layer_string(H.dec_blocks)
        for idx, (res, mixin) in enumerate(blocks):
            dec_blocks.append(DecBlock(H, res, mixin, n_blocks=len(blocks)))
            resos.add(res)
        self.resolutions = sorted(resos)
        self.dec_blocks = nn.ModuleList(dec_blocks)
        first_res = self.resolutions[0]
        self.constant = nn.Parameter(torch.randn(1, self.widths[first_res], first_res, first_res))
        self.resnet = get_1x1(H.width, H.image_channels)
        self.gain = nn.Parameter(torch.ones(1, H.image_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, H.image_channels, 1, 1))

    def forward(self, latent_code, spatial_noise, input_is_w=False):
        if not input_is_w:
            w = self.mapping_network(latent_code)[0]
        else:
            w = latent_code
        
        x = self.constant.repeat(latent_code.shape[0], 1, 1, 1)

        for idx, block in enumerate(self.dec_blocks):
            noise = None
            x = block(x, w, noise)
        x = self.resnet(x)
        x = self.gain * x + self.bias
        return x


class IMLE(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.dci_db = None
        self.img_size = H.image_size
        self.latent_channels = 4
        # self.decoder = Decoder(H)
        # self.decoder = FixedUNet32()

        # self.decoder = DiT_S_2()
        # self.decoder = Decoder(H)
        in_ch = 4
        cond_ch = in_ch
        total_in = in_ch + cond_ch  # 8

        self.decoder = UNetModelWrapper(
            dim=(total_in, H.image_size // 8, H.image_size // 8),
            num_res_blocks=2,
            num_channels=128,
            channel_mult=[1, 2, 2, 2],
            num_heads=4,
            num_head_channels=64,
            attention_resolutions="16",
            dropout=0.1,
            # concat-only: disable internal conditioning paths
            cond_channels=0,
            cond_concat=False,
            cond_resblock_mode="none",
        )



    def forward(self, latents, spatial_noise=None, input_is_w=False, condition_data=None, condition_indices=None, return_condition=False):
        if latents.ndim != 4:
            batch_size = latents.shape[0]
            latent_side = self.img_size // 8
            latents = latents.reshape(batch_size, self.latent_channels, latent_side, latent_side)
        x_cat = torch.cat([latents, condition_data], dim=1)
        out = self.decoder(x_cat)
        if return_condition:
            # Return the raw condition_data and optional indices for debugging
            return out, { 'condition_data': condition_data, 'condition_indices': condition_indices }
        return out