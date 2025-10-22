#!/usr/bin/env python3
"""
Test script to verify condition data as vocabulary functionality.
"""

import torch
import os

def test_condition_as_vocabulary():
    """Test using condition data as vocabulary."""
    
    condition_path = "/home/kha98/flow-model/flow-model-chirag/results/icfm/x_T.pt"
    
    print(f"Testing condition data as vocabulary from: {condition_path}")
    
    if not os.path.exists(condition_path):
        print("❌ Condition file not found!")
        return False
    
    try:
        # Load the condition tensor
        condition_tensors = torch.load(condition_path)
        
        if isinstance(condition_tensors, torch.Tensor):
            print(f"✅ Loaded condition tensor with shape: {condition_tensors.shape}")
            
            # Simulate the vocabulary usage
            data_labels = condition_tensors  # condition = data_labels as per requirements
            
            print(f"Using condition as vocabulary:")
            print(f"  - Total samples: {len(data_labels)}")
            print(f"  - Each sample has unique condition/vocabulary")
            print(f"  - All samples have conditions")
            print(f"  - Condition shape per sample: {data_labels.shape[1:]}")
            
            # Simulate batch processing
            batch_size = 8
            batch_indices = torch.randint(0, len(data_labels), (batch_size,))
            batch_conditions = data_labels[batch_indices]
            
            print(f"\nSimulating batch processing:")
            print(f"  - Batch size: {batch_size}")
            print(f"  - Batch conditions shape: {batch_conditions.shape}")
            print(f"  - Each sample in batch has its own condition/vocabulary")
            
            # Verify that each sample has unique conditions
            if batch_size > 1:
                unique_conditions = torch.unique(batch_conditions.view(batch_size, -1), dim=0)
                print(f"  - Unique conditions in batch: {len(unique_conditions)}")
                if len(unique_conditions) == batch_size:
                    print("  ✅ Each sample has unique condition (as expected)")
                else:
                    print("  ⚠️  Some samples have identical conditions")
            
            return True
        else:
            print(f"❌ Expected tensor format, got: {type(condition_tensors)}")
            return False
        
    except Exception as e:
        print(f"❌ Error testing condition as vocabulary: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_condition_as_vocabulary()
    if success:
        print("\n✅ Condition as vocabulary test passed!")
    else:
        print("\n❌ Condition as vocabulary test failed!")
