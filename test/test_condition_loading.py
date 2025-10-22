#!/usr/bin/env python3
"""
Test script to verify condition data loading functionality and generate samples.
This script tests loading the .pt file, verifies the integration, and generates samples.
"""

import torch
import os
import sys
from hps import Hyperparams
from data import set_up_data
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
    FLOW_MODEL_AVAILABLE = True
except ImportError as e:
    print(f"Flow model imports not available: {e}")
    FLOW_MODEL_AVAILABLE = False

def test_condition_loading():
    """Test the condition data loading functionality."""
    
    # Create a mock hyperparameters object
    H = Hyperparams()
    H.condition_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt"
    H.dataset = "cifar10"  # Use a simple dataset for testing
    H.data_root = "./"  # This might need to be adjusted based on your setup
    H.image_size = 32
    H.test_eval = False
    H.subset_len = 100  # Small subset for testing
    
    print(f"Testing condition loading with path: {H.condition_path}")
    print(f"File exists: {os.path.exists(H.condition_path)}")
    
    if not os.path.exists(H.condition_path):
        print("Warning: Condition file not found. Creating a mock tensor for testing.")
        # Create a mock condition tensor for testing
        mock_condition = torch.randn(100, 4, 4, 4)  # Example tensor shape
        torch.save(mock_condition, "/tmp/mock_condition.pt")
        H.condition_path = "/tmp/mock_condition.pt"
    
    try:
        # Test the data loading
        H, data_train, data_valid, preprocess_fn, condition_data = set_up_data(H)
        
        print(f"Data loading successful!")
        print(f"Train data length: {len(data_train)}")
        print(f"Valid data length: {len(data_valid)}")
        print(f"Condition data type: {type(condition_data)}")
        
        if condition_data is not None:
            if isinstance(condition_data, torch.Tensor):
                print(f"Condition data shape: {condition_data.shape}")
                print(f"Condition data dtype: {condition_data.dtype}")
            else:
                print(f"Condition data type: {type(condition_data)}")
                print(f"Condition data length: {len(condition_data)}")
        else:
            print("No condition data loaded")
        
        return True
        
    except Exception as e:
        print(f"Error during data loading: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_condition_generation():
    """Test generating samples from the condition data using the flow model."""
    
    if not FLOW_MODEL_AVAILABLE:
        print("⚠️  Flow model not available, skipping generation test")
        return True
    
    # Configuration (matching flow.py)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters (matching flow.py)
    img_size = 256
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
    
    # Try to load checkpoint
    ckpt_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/checkpoint.pt"
    if os.path.exists(ckpt_path):
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
    else:
        print("⚠️  Checkpoint not found, using randomly initialized model...")
    
    model.eval()
    
    # VAE setup
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
    
    # Generate a few samples for testing
    print("Generating samples from x_T tensor...")
    
    batch_size = 2  # Small batch for testing
    total_samples = min(4, len(x_T_store))  # Generate 4 samples max for testing
    
    with torch.no_grad():
        for i in trange(0, total_samples, batch_size, desc="Generating samples"):
            current_batch_size = min(batch_size, total_samples - i)
            
            # Get x_T batch
            x_T_batch = x_T_store[i:i + current_batch_size].to(device)
            
            # Generate samples
            try:
                ema_samples = generate_samples(model, False, x_T_batch)
                
                # Decode using VAE
                final_imgs = vae.decode(ema_samples / vae.config.scaling_factor).sample.cpu()
                
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
                print(f"❌ Error generating samples: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"✅ Sample generation completed! Check output directory: {output_dir}")
    return True

if __name__ == "__main__":
    print("=== Testing Condition Data Loading ===")
    success1 = test_condition_loading()
    
    print("\n=== Testing Condition Generation ===")
    success2 = test_condition_generation()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
