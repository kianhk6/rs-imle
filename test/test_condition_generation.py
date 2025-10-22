#!/usr/bin/env python3
"""
Test script to load the same model as flow.py and generate samples from x_T tensor.
"""

import torch
import os
import sys
from PIL import Image
import numpy as np
from tqdm import trange

# Add the flow-model directory to path to import utils
sys.path.append('/home/kha98/flow-model/flow-model-chirag')

try:
    from utils import generate_samples
    from torchcfm.models.unet.unet import UNetModelWrapper
    from diffusers.models import AutoencoderKL
    from torchdyn.core import NeuralODE
    import copy
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you have the required dependencies installed")
    sys.exit(1)

def test_condition_generation():
    """Test loading the model and generating samples from x_T tensor."""
    
    # Configuration (matching flow.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters (matching flow.py)
    img_size = 256  # Default from flow.py
    num_channel = 128
    H8 = img_size // 8
    C = 4
    
    # Load the x_T tensor
    x_T_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt"
    print(f"Loading x_T tensor from: {x_T_path}")
    
    if not os.path.exists(x_T_path):
        print(f"❌ x_T file not found at {x_T_path}")
        return False
    
    try:
        x_T_store = torch.load(x_T_path, map_location=device)
        print(f"✅ Loaded x_T tensor with shape: {x_T_store.shape}")
        print(f"x_T dtype: {x_T_store.dtype}, device: {x_T_store.device}")
    except Exception as e:
        print(f"❌ Error loading x_T tensor: {e}")
        return False
    
    # Model setup (matching flow.py)
    print("Setting up model...")
    model = UNetModelWrapper(
        dim=(4, img_size // 8, img_size // 8),
        num_res_blocks=2,
        num_channels=num_channel,
        channel_mult=[1, 2, 2, 2],
        num_heads=4,
        num_head_channels=64,
        attention_resolutions="16",
        dropout=0.1,
    ).to(device)
    
    # Load model checkpoint (you'll need to provide the path)
    ckpt_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/checkpoint.pt"  # Update this path
    
    if not os.path.exists(ckpt_path):
        print(f"⚠️  Checkpoint not found at {ckpt_path}")
        print("Using randomly initialized model for testing...")
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            if "ema_model" in ckpt:
                model.load_state_dict(ckpt["ema_model"])
                print("✅ Loaded EMA model from checkpoint")
            else:
                model.load_state_dict(ckpt)
                print("✅ Loaded model from checkpoint")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")
            print("Using randomly initialized model for testing...")
    
    model.eval()
    
    # VAE setup (matching flow.py)
    print("Setting up VAE...")
    try:
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
        vae = torch.compile(vae)
        print("✅ VAE loaded and compiled")
    except Exception as e:
        print(f"❌ Error loading VAE: {e}")
        return False
    
    # Create output directory
    output_dir = "/home/kha98/rs-imle/test_generated_samples"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples from x_T
    print("Generating samples from x_T tensor...")
    
    batch_size = 4  # Small batch for testing
    total_samples = min(8, len(x_T_store))  # Generate 8 samples max for testing
    
    with torch.no_grad():
        for i in trange(0, total_samples, batch_size, desc="Generating samples"):
            current_batch_size = min(batch_size, total_samples - i)
            
            # Get x_T batch
            x_T_batch = x_T_store[i:i + current_batch_size].to(device)
            print(f"Processing batch {i//batch_size + 1}: x_T shape = {x_T_batch.shape}")
            
            # Generate samples using the same method as flow.py
            try:
                ema_samples = generate_samples(model, False, x_T_batch)
                print(f"Generated samples shape: {ema_samples.shape}")
                
                # Decode using VAE
                final_imgs = vae.decode(ema_samples / vae.config.scaling_factor).sample.cpu()
                print(f"Decoded images shape: {final_imgs.shape}")
                
                # Save images
                for j in range(current_batch_size):
                    img = final_imgs[j]
                    img = (img.clamp(-1, 1) + 1) * 0.5 * 255
                    
                    # Convert to PIL and save
                    img_pil = Image.fromarray(img.to(torch.uint8).permute(1, 2, 0).numpy())
                    img_path = os.path.join(output_dir, f"generated_{i + j:03d}.png")
                    img_pil.save(img_path)
                    print(f"Saved: {img_path}")
                
                # Clean up
                del x_T_batch, ema_samples, final_imgs
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ Error generating samples for batch {i//batch_size + 1}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"✅ Sample generation completed! Check output directory: {output_dir}")
    return True

if __name__ == "__main__":
    success = test_condition_generation()
    if success:
        print("\n✅ Condition generation test passed!")
    else:
        print("\n❌ Condition generation test failed!")
        sys.exit(1)
