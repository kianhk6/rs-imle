#!/usr/bin/env python3
import argparse
import importlib.machinery, importlib.util, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from diffusers.models import AutoencoderKL

# ------------------------------
# Utilities
# ------------------------------
def load_module_from_path(module_name: str, file_path: str):
    loader = importlib.machinery.SourceFileLoader(module_name, file_path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    sys.modules[module_name] = module
    return module

def find_first_class(mod, substrings):
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and any(s.lower() in name.lower() for s in substrings):
            return obj
    return None

# ------------------------------
# External VAE for UNet (exact config)
# ------------------------------
def load_unet_vae_exact(device: torch.device):
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
    # match your reference: compile the decoder
    try:
        vae = torch.compile(vae)
    except Exception:
        # torch.compile may be unavailable; still proceed
        pass
    for p in vae.parameters():
        p.requires_grad = False
    scaling = float(getattr(vae.config, "scaling_factor", 0.18215))
    return vae, scaling

# ------------------------------
# Train steps (no warmup)
# ------------------------------
def unet_train_step(unet, vae, scaling, x_latent, optimizer, criterion):
    optimizer.zero_grad(set_to_none=True)
    lat_out = unet(x_latent)                              # (B,4,H,W)
    img = vae.decode(lat_out / scaling).sample            # exact: divide by vae.config.scaling_factor
    loss = criterion(img, torch.randn_like(img))
    loss.backward()
    optimizer.step()
    return loss.item()

def dit_train_step(dit, x_latent, optimizer, criterion, flatten_input=False):
    optimizer.zero_grad(set_to_none=True)
    x_in = x_latent.view(x_latent.size(0), -1) if flatten_input else x_latent
    out = dit(x_in)
    if isinstance(out, dict) and "sample" in out:
        out = out["sample"]
    loss = criterion(out, torch.randn_like(out))
    loss.backward()
    optimizer.step()
    return loss.item()

def benchmark_unet(unet, vae, scaling, x_latent, iters, device):
    criterion = nn.MSELoss()
    opt = optim.Adam(unet.parameters(), lr=1e-4)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        unet_train_step(unet, vae, scaling, x_latent, opt, criterion)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return sum(times)/len(times), min(times), max(times)

def benchmark_dit(dit, x_latent, iters, device, flatten_input):
    criterion = nn.MSELoss()
    opt = optim.Adam(dit.parameters(), lr=1e-4)
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        dit_train_step(dit, x_latent, opt, criterion, flatten_input=flatten_input)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return sum(times)/len(times), min(times), max(times)

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="Train-step speed: UNet (external sd-vae-ft-mse) vs DiT (as-is)")
    p.add_argument("--unet_path", type=str, required=True)
    p.add_argument("--dit_path",  type=str, required=True)
    p.add_argument("--unet_class", type=str, default="")
    p.add_argument("--dit_class",  type=str, default="")
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--hw",    type=int, default=32)   # latent H=W
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--dit_flatten", action="store_true",
                   help="If your DiT expects (B, 4*H*W), set this.")
    args = p.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    dev = torch.device(device)

    # UNet
    unet_mod = load_module_from_path("user_unet", str(Path(args.unet_path).resolve()))
    UNetClass = (getattr(unet_mod, args.unet_class, None)
                 or getattr(unet_mod, "FixedUNet32", None)
                 or find_first_class(unet_mod, ["unet"]))
    if UNetClass is None:
        raise RuntimeError(f"Could not find a UNet class in {args.unet_path}")
    unet = UNetClass().to(dev).train()

    # DiT (as-is)
    dit_mod = load_module_from_path("user_dit", str(Path(args.dit_path).resolve()))
    DiTClass = (getattr(dit_mod, args.dit_class, None)
                or getattr(dit_mod, "DiT_S_2", None)
                or find_first_class(dit_mod, ["dit"]))
    if DiTClass is None:
        raise RuntimeError(f"Could not find a DiT class in {args.dit_path}")
    dit = DiTClass().to(dev).train()

    # External decoder ONLY for UNet — exact config
    vae, scaling = load_unet_vae_exact(dev)

    # Synthetic latent input
    x = torch.randn(args.batch, 4, args.hw, args.hw, device=dev)

    # Benchmarks (no warmup)
    u_avg, u_min, u_max = benchmark_unet(unet, vae, scaling, x, args.iters, device)
    d_avg, d_min, d_max = benchmark_dit(dit, x, args.iters, device, args.dit_flatten)

    print("\n=== Training Step Benchmark Results (no warmup) ===")
    print(f"UNet (external sd-vae-ft-mse): avg={u_avg:.3f} ms, min={u_min:.3f}, max={u_max:.3f}")
    print(f"DiT  (as-is):                  avg={d_avg:.3f} ms, min={d_min:.3f}, max={d_max:.3f}")
    print("===================================================\n")

if __name__ == "__main__":
    main()
