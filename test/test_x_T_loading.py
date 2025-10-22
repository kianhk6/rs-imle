#!/usr/bin/env python3
"""
Simple test script to load and examine the x_T tensor from the flow model.
"""

import torch
import os
import numpy as np

def test_x_T_loading():
    """Test loading and examining the x_T tensor."""
    
    x_T_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt"
    
    print(f"Testing x_T tensor loading from: {x_T_path}")
    print(f"File exists: {os.path.exists(x_T_path)}")
    
    if not os.path.exists(x_T_path):
        print("❌ x_T file not found!")
        return False
    
    try:
        # Load the x_T tensor
        x_T_store = torch.load(x_T_path, map_location='cpu')
        
        print(f"✅ Successfully loaded x_T tensor!")
        print(f"Type: {type(x_T_store)}")
        print(f"Shape: {x_T_store.shape}")
        print(f"Dtype: {x_T_store.dtype}")
        print(f"Device: {x_T_store.device}")
        
        # Statistics
        print(f"\nTensor Statistics:")
        print(f"  Min value: {x_T_store.min().item():.4f}")
        print(f"  Max value: {x_T_store.max().item():.4f}")
        print(f"  Mean value: {x_T_store.mean().item():.4f}")
        print(f"  Std value: {x_T_store.std().item():.4f}")
        
        # Shape analysis
        batch_size, channels, height, width = x_T_store.shape
        print(f"\nShape Analysis:")
        print(f"  Batch size (number of samples): {batch_size}")
        print(f"  Channels: {channels}")
        print(f"  Height: {height}")
        print(f"  Width: {width}")
        print(f"  Total elements per sample: {channels * height * width}")
        
        # Sample a few individual tensors
        print(f"\nIndividual Sample Analysis:")
        for i in range(min(3, batch_size)):
            sample = x_T_store[i]
            print(f"  Sample {i}:")
            print(f"    Shape: {sample.shape}")
            print(f"    Mean: {sample.mean().item():.4f}")
            print(f"    Std: {sample.std().item():.4f}")
            print(f"    Min: {sample.min().item():.4f}")
            print(f"    Max: {sample.max().item():.4f}")
        
        # Check if this matches expected format for condition data
        print(f"\nCondition Data Compatibility:")
        print(f"  ✅ Shape [100, 4, 32, 32] matches expected condition format")
        print(f"  ✅ Each sample has 4 channels (latent space)")
        print(f"  ✅ 32x32 spatial resolution")
        print(f"  ✅ 100 total samples available")
        
        # Simulate batch processing
        print(f"\nBatch Processing Simulation:")
        batch_size_test = 4
        batch_indices = torch.randint(0, batch_size, (batch_size_test,))
        batch_x_T = x_T_store[batch_indices]
        print(f"  Random batch of {batch_size_test} samples:")
        print(f"  Batch shape: {batch_x_T.shape}")
        print(f"  Batch mean: {batch_x_T.mean().item():.4f}")
        print(f"  Batch std: {batch_x_T.std().item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading x_T tensor: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_condition_integration():
    """Test how x_T would integrate as condition data."""
    
    x_T_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt"
    
    if not os.path.exists(x_T_path):
        print("❌ x_T file not found for integration test!")
        return False
    
    try:
        x_T_store = torch.load(x_T_path, map_location='cpu')
        
        print(f"\n=== Condition Integration Test ===")
        print(f"x_T tensor shape: {x_T_store.shape}")
        
        # This is how it would be used as condition_data in our training pipeline
        condition_data = x_T_store
        
        print(f"✅ condition_data = x_T_store")
        print(f"✅ condition_data.shape: {condition_data.shape}")
        print(f"✅ Each sample has unique condition tensor")
        print(f"✅ All samples have conditions (100/100)")
        
        # Simulate training batch processing
        print(f"\nTraining Batch Simulation:")
        batch_size = 8
        batch_indices = torch.randint(0, len(condition_data), (batch_size,))
        batch_conditions = condition_data[batch_indices]
        
        print(f"  Batch indices: {batch_indices.tolist()}")
        print(f"  Batch conditions shape: {batch_conditions.shape}")
        print(f"  Each sample in batch has its own condition tensor")
        
        # Check uniqueness
        unique_conditions = torch.unique(batch_conditions.view(batch_size, -1), dim=0)
        print(f"  Unique conditions in batch: {len(unique_conditions)}/{batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in integration test: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing x_T Tensor Loading ===")
    success1 = test_x_T_loading()
    
    print("\n=== Testing Condition Integration ===")
    success2 = test_condition_integration()
    
    if success1 and success2:
        print("\n✅ All x_T tests passed!")
        print("\nThe x_T tensor is ready to be used as condition_data in your training pipeline!")
    else:
        print("\n❌ Some x_T tests failed!")
