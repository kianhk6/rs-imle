#!/usr/bin/env python3
"""
Simple test script to verify condition tensor loading.
"""

import torch
import os

def test_condition_tensor_loading():
    """Test loading the condition tensor file."""
    
    condition_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt"
    
    print(f"Testing condition tensor loading from: {condition_path}")
    print(f"File exists: {os.path.exists(condition_path)}")
    
    if not os.path.exists(condition_path):
        print("❌ Condition file not found!")
        return False
    
    try:
        # Load the condition tensor
        condition_tensors = torch.load(condition_path)
        
        print(f"✅ Successfully loaded condition tensor!")
        print(f"Type: {type(condition_tensors)}")
        
        if isinstance(condition_tensors, torch.Tensor):
            print(f"Tensor shape: {condition_tensors.shape}")
            print(f"Tensor dtype: {condition_tensors.dtype}")
            print(f"Tensor device: {condition_tensors.device}")
            print(f"Tensor min value: {condition_tensors.min().item():.4f}")
            print(f"Tensor max value: {condition_tensors.max().item():.4f}")
            print(f"Tensor mean value: {condition_tensors.mean().item():.4f}")
            print(f"Tensor std value: {condition_tensors.std().item():.4f}")
        elif isinstance(condition_tensors, (list, tuple)):
            print(f"List/tuple length: {len(condition_tensors)}")
            for i, tensor in enumerate(condition_tensors):
                if isinstance(tensor, torch.Tensor):
                    print(f"  Tensor {i}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
                else:
                    print(f"  Item {i}: type={type(tensor)}")
        else:
            print(f"Unexpected type: {type(condition_tensors)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading condition tensor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_condition_tensor_loading()
    if success:
        print("\n✅ Condition tensor loading test passed!")
    else:
        print("\n❌ Condition tensor loading test failed!")
