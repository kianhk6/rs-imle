# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from diffusers.models import AutoencoderKL

# The primary purpose of writing it 1+scale is to ensure that when the learned scale is zero, 
# the multiplicative factor is 1 (i.e. an identity transformation). 
# This means that if no scaling is learned (scale = 0), the input is left unchanged.
# def modulate(x, shift, scale):
#     return x * scale + shift
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations and calculates the element-wise
    running mean tensor over all function calls. It prints the full tensor every 100 (or 1000)
    calls. In this updated version, instead of receiving t as input, we use a constant tensor.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        
        # Initialize variables for accumulating t_freq tensors and tracking statistics.
        self.num_calls = 0
        self.running_sum_tensor = None
        self.curr_max = float('-inf')

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D or 2-D Tensor of indices.
        :param dim: the dimension of the output embedding.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: a Tensor of shape (..., dim) of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32,  device="cuda") / half
        )
        # Ensure t is float and reshape for broadcasting: (N, 1)
        args = t.float().unsqueeze(-1) * freqs  # result shape: (N, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[..., :1])], dim=-1)
        return embedding

    def forward(self, t):
        # Use the constant tensor as the "t" input.
        # t = torch.tensor([499.5, 499.5, 499.5, 499.5],  device="cuda")
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        
        # Pass the timestep embeddings through the MLP.
        t_emb = self.mlp(t_freq)
        return t_emb


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        # core layers
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn  = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hid    = int(hidden_size * mlp_ratio)
        self.mlp   = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hid,
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(),
                                              nn.Linear(hidden_size, 6 * hidden_size, bias=True))

        # the eight “flags” where we measure variance
        self.stat_keys = [
            "prior_x", "norm1_x", "modulated_msa", "attn_out", "x_after_msa",
            "norm2_x", "modulated_mlp", "mlp_out",   "final_x"
        ]

        # initialize accumulators as Python attrs
        for flag in self.stat_keys:
            setattr(self, f"total_sum_{flag}",    None)   # will be a Tensor (features,)
            setattr(self, f"total_sum_sq_{flag}", None)   # will be a Tensor (features,)
            setattr(self, f"total_images_{flag}", 0)      # scalar
        
        self.prev_mean_var = None

    def _update_stats(self, x: torch.Tensor, flag: str):
        """
        x: (B, num_patches, embed_dim)
        """
        B = x.shape[0]
        flat = x.reshape(B, -1)              # → (B, features)
        sum_feat    = flat.sum(dim=0)        # (features,)
        sum_sq_feat = (flat * flat).sum(dim=0)

        # first batch for this flag?
        tot = getattr(self, f"total_sum_{flag}")
        if tot is None:
            # detach & clone so we don’t track grads
            setattr(self, f"total_sum_{flag}",    sum_feat.detach().clone())
            setattr(self, f"total_sum_sq_{flag}", sum_sq_feat.detach().clone())
        else:
            getattr(self, f"total_sum_{flag}").add_(sum_feat.detach())
            getattr(self, f"total_sum_sq_{flag}").add_(sum_sq_feat.detach())

        # increment image count
        cur = getattr(self, f"total_images_{flag}")
        setattr(self, f"total_images_{flag}", cur + B)
    
    def _output_variance(self, flag: str):
        S  = getattr(self, f"total_sum_{flag}")     # Tensor(features,)
        S2 = getattr(self, f"total_sum_sq_{flag}")  # Tensor(features,)
        N  = getattr(self, f"total_images_{flag}")  # scalar

        # compute per-feature population moments
        mean_feat    = S  / N
        mean_sq_feat = S2 / N

        var_feat = mean_sq_feat - mean_feat * mean_feat
        if(N % 100 == 0):
            print(f"{flag}: ⟨Var⟩ over {N} images = {var_feat.mean().item():.8f}")
        mean_var = var_feat.mean().item()

        return mean_var
    
    def _report_layer(self, tensor, flag, prev_name, prev_var):
        """
        • updates running statistics for `flag`
        • prints in the form:
        "<prev: > <prev_var> -> <flag: > <curr_var> (±xx.xx%)"
        • returns new (name, var) pair for chaining
        """
        self._update_stats(tensor, flag)
        curr_var = self._output_variance(flag)
        N  = getattr(self, f"total_images_{flag}")  # scalar

        if(N % 1024 == 0):
            if prev_name is None:                       # first layer
                print(f"{flag}: {curr_var:.6e}")
            else:
                pct = (curr_var - prev_var) / prev_var * 100 if prev_var != 0 else float('nan')
                print(f"{prev_name}: {prev_var:.6e} -> {flag}: {curr_var:.6e} ({pct:+.2f}%)")

        return flag, curr_var

    def forward(self, x, c, isSample=False):
        prev_var  = None            # numeric value
        prev_name = None            # string label

        # ---- collect stats for the raw input (optional) ----
        if isSample:
            self._update_stats(x, "prior_x")
            curr_var = self._output_variance("prior_x")      # scalar
            # print(f"prior_x: {curr_var:.6e}")                # first layer: no % yet
            prev_name, prev_var = "prior_x", curr_var

        # -------- get shift/scale/gate params --------
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)

        # ---------------------------------------------------------------- MSA branch
        # 1) LayerNorm
        norm1_x = self.norm1(x)
        if isSample:
            prev_name, prev_var = self._report_layer(norm1_x, "norm1_x", prev_name, prev_var)

        # 2) modulation
        mod_msa = modulate(norm1_x, shift_msa, scale_msa)
        if isSample:
            prev_name, prev_var = self._report_layer(mod_msa, "modulated_msa", prev_name, prev_var)

        # 3) self-attention
        attn_out = self.attn(mod_msa)
        if isSample:
            prev_name, prev_var = self._report_layer(attn_out, "attn_out", prev_name, prev_var)

        # 4) add attn residual
        x = x + gate_msa.unsqueeze(1) * attn_out
        if isSample:
            prev_name, prev_var = self._report_layer(x, "x_after_msa", prev_name, prev_var)

        # ---------------------------------------------------------------- MLP branch
        # 5) second LayerNorm
        norm2_x = self.norm2(x)
        if isSample:
            prev_name, prev_var = self._report_layer(norm2_x, "norm2_x", prev_name, prev_var)

        # 6) modulation
        mod_mlp = modulate(norm2_x, shift_mlp, scale_mlp)
        if isSample:
            prev_name, prev_var = self._report_layer(mod_mlp, "modulated_mlp", prev_name, prev_var)

        # 7) MLP
        mlp_out = self.mlp(mod_mlp)
        if isSample:
            prev_name, prev_var = self._report_layer(mlp_out, "mlp_out", prev_name, prev_var)

        # 8) add MLP residual  ➜ final_x
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        if isSample:
            self._report_layer(x, "final_x", prev_name, prev_var)

        return x


class FinalLayer(nn.Module):
    """
    Final DiT layer with variance tracking and
    formatted prints like “prior_x -> norm_x (+XX.XX%)”.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final       = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear           = nn.Linear(hidden_size, patch_size*patch_size*out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias=True)
        )

        self.stat_keys = ["init_x", "norm_x", "modulated_x", "linear_out"]
        for flag in self.stat_keys:
            setattr(self, f"total_sum_{flag}",    None)
            setattr(self, f"total_sum_sq_{flag}", None)
            setattr(self, f"total_images_{flag}", 0)

    def _update_stats(self, tensor: torch.Tensor, flag: str):
        B    = tensor.shape[0]
        flat = tensor.reshape(B, -1)
        sum_feat    = flat.sum(dim=0)
        sum_sq_feat = (flat * flat).sum(dim=0)

        tot = getattr(self, f"total_sum_{flag}")
        if tot is None:
            setattr(self, f"total_sum_{flag}",    sum_feat.detach().clone())
            setattr(self, f"total_sum_sq_{flag}", sum_sq_feat.detach().clone())
        else:
            getattr(self, f"total_sum_{flag}").add_(sum_feat.detach())
            getattr(self, f"total_sum_sq_{flag}").add_(sum_sq_feat.detach())

        cur = getattr(self, f"total_images_{flag}")
        setattr(self, f"total_images_{flag}", cur + B)

    def _get_mean_variance(self, flag: str) -> float:
        S  = getattr(self, f"total_sum_{flag}")
        S2 = getattr(self, f"total_sum_sq_{flag}")
        N  = getattr(self, f"total_images_{flag}")

        mean_feat    = S  / N
        mean_sq_feat = S2 / N
        var_feat     = mean_sq_feat - mean_feat * mean_feat
        return var_feat.mean().item()

    def forward(self, x, c, sampling=False):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)

        if not sampling:
            return self.linear(modulate(self.norm_final(x), *self.adaLN_modulation(c).chunk(2, dim=1)))

        # 0) init_x
        self._update_stats(x, "init_x")
        init_var = self._get_mean_variance("init_x")

        # 1) norm_x
        norm_x = self.norm_final(x)
        self._update_stats(norm_x, "norm_x")
        norm_var = self._get_mean_variance("norm_x")
        pct = (norm_var - init_var) / init_var * 100 if init_var != 0 else float('nan')
        print(f"init_x:    {init_var:.6e} -> norm_x:    {norm_var:.6e} ({pct:+.2f}%)")

        # 2) modulated_x
        mod_x = modulate(norm_x, shift, scale)
        self._update_stats(mod_x, "modulated_x")
        mod_var = self._get_mean_variance("modulated_x")
        pct = (mod_var - norm_var) / norm_var * 100 if norm_var != 0 else float('nan')

        # 3) linear_out
        out = self.linear(mod_x)
        self._update_stats(out, "linear_out")
        lin_var = self._get_mean_variance("linear_out")
        pct = (lin_var - mod_var) / mod_var * 100 if mod_var != 0 else float('nan')
        print(f"modulated_x:{mod_var:.6e} -> linear_out:  {lin_var:.6e} ({pct:+.2f}%)")

        return out


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.0,
        num_classes=1000,
        learn_sigma=False,
        H=None  
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.H = H 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)        
        self.y_embedder = None

        num_patches = self.x_embedder.num_patches
        self.hidden_size = hidden_size
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        # self.norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(self.device)
        # Freeze all parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        self.initialize_weights()

        # Set up a learnable time constant if H.learnable_t is True.
        if self.H is not None and self.H.learnable_t == True:
            # # print("Learnable t parameter is set to True.")
            # Create a learnable parameter initialized to 499.5.
            self.t_parameter = nn.Parameter(torch.tensor(499.5, device=self.device))
        else:
            self.t_parameter = None

        self.total_sum_unpatch   = None   # 1-D tensor (C*H*W,)
        self.total_sum_sq_unpatch= None
        self.total_imgs_unpatch  = 0      # scalar

        self.total_sum_decoded   = None   # 1-D tensor (3*H*W,)
        self.total_sum_sq_decoded= None
        self.total_imgs_decoded  = 0

    @staticmethod
    def _update_global_stats(tensor, total_sum, total_sum_sq, total_N):
        """
        tensor : (B, C, H, W)
        Updates the three running variables *in place* and returns them.
        """
        B = tensor.shape[0]
        flat = tensor.reshape(B, -1)                       # (B, features)
        sum_feat    = flat.sum(dim=0)                      # (features,)
        sum_sq_feat = (flat * flat).sum(dim=0)             # (features,)

        if total_sum is None:                  # first call
            total_sum    = sum_feat.detach().clone()
            total_sum_sq = sum_sq_feat.detach().clone()
        else:
            total_sum    += sum_feat.detach()
            total_sum_sq += sum_sq_feat.detach()

        total_N += B
        return total_sum, total_sum_sq, total_N

    @staticmethod
    def _mean_variance(total_sum, total_sum_sq, total_N):
        mean     = total_sum / total_N
        mean_sq  = total_sum_sq / total_N
        var_feat = mean_sq - mean * mean
        return var_feat.mean().item()          # scalar
    
    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

        nn.init.xavier_uniform_(self.final_layer.linear.weight, gain=0.2)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        # # print(f"Unpatchified images shape: {imgs.shape}")
        return imgs
    
    def forward(self, x, sampling=False, inle_stuff=None, t=None, y=None,):
        """
        Forward pass of DiT without timestep embeddings.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images).
        """
        # Spatial embedding with positional information

        # Assuming cur_latents is of shape (batch, flattened_size)
        batch, flattened_size = x.shape
        channels = 4
        height = width = int((flattened_size // channels) ** 0.5)  # Calculate spatial dimensions
        B, D = x.shape  

        # Decide on the timestep constant based on whether a learnable parameter has been set.
        if self.t_parameter is not None:
            # Use the learnable parameter and expand it to create a tensor of shape (B,).
            t_val = self.t_parameter.expand(B)
        else:
            t_val = torch.randint(0, 999, (x.shape[0],), device=x.device)

        # # print(self.H.elementwise_affine)c
        # # print(x)
        # # print(t)
        c = self.t_embedder(t_val)

        # Ensure compatibility
        assert flattened_size == channels * height * width, "Flattened size must be divisible by channels"

        # Reshape cur_latents
        x = x.view(batch, channels, height, width)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2       

 
        # x = self.x_embedder(x) # (N, T, D), where T = H * W / patch_size ** 2        
        # Pass through transformer blocks
        i = 0
        for block in self.blocks:
            if sampling:
                print(f"{i}:")
            x = block(x, c, sampling)  # Blocks are adjusted to handle c=None
            i += 1
            if sampling:
                print("\n\n")

        x = self.final_layer(x, c, sampling) * 2.75  # Final layer adjusted to handle c=None

        x = self.unpatchify(x)  # Convert back to spatial format (N, out_channels, H, W)
        # # Decode the latent points to images

        if sampling:                          
            (self.total_sum_unpatch,
            self.total_sum_sq_unpatch,
            self.total_imgs_unpatch) = self._update_global_stats(
                x,
                self.total_sum_unpatch,
                self.total_sum_sq_unpatch,
                self.total_imgs_unpatch
            )
            var_lat = self._mean_variance(
                self.total_sum_unpatch,
                self.total_sum_sq_unpatch,
                self.total_imgs_unpatch
            )
            print(f"[unpatchify]  Var = {var_lat:.6e}  (N={self.total_imgs_unpatch})")

            
        decoded_images = self.vae.decode(x / 0.18215).sample

        if sampling:                                  # --- variance for decoded ---
            (self.total_sum_decoded,
            self.total_sum_sq_decoded,
            self.total_imgs_decoded) = self._update_global_stats(
                decoded_images,
                self.total_sum_decoded,
                self.total_sum_sq_decoded,
                self.total_imgs_decoded
            )
            var_dec = self._mean_variance(
                self.total_sum_decoded,
                self.total_sum_sq_decoded,
                self.total_imgs_decoded
            )
            # percentage change from latent → decoded

            pct = (var_dec - var_lat) / var_lat * 100 if var_lat != 0 else float('nan')
            print(f"[decoded]     Var = {var_dec:.6e}  (Δ vs latent = {pct:+.2f}%)")
        
        return decoded_images


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(H=None, **kwargs):
    return DiT(
        depth=12, 
        hidden_size=384, 
        patch_size=2, 
        num_heads=6, 
        H=H,            # use the function parameter, not a global
        **kwargs
    )

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
