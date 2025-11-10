"""
Comprehensive seeding utilities for reproducibility.

This module provides functions to set random seeds across all libraries
used in the project while maintaining performance.
"""

import os
import random
import numpy as np
import torch


def seed_everything(seed=0, deterministic_mode=False):
    """
    Set random seed for all libraries to ensure reproducibility.
    
    Args:
        seed (int): Random seed value
        deterministic_mode (bool): If True, enables full determinism (slower but more reproducible)
                                   If False, keeps cudnn.benchmark=True for speed
    
    Returns:
        None
    """
    # Python's built-in random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch CPU
    torch.manual_seed(seed)
    
    # PyTorch CUDA - all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Python hash seed (for dictionaries, sets)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CuDNN settings
    if deterministic_mode:
        # Full determinism - slower but completely reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"⚠️  Deterministic mode enabled (slower). cudnn.benchmark=False")
    else:
        # Fast mode - good reproducibility but not 100% deterministic due to cudnn
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"✓ Fast mode with seeding. cudnn.benchmark=True (for speed)")
    
    print(f"✓ Seeded all RNGs with seed={seed}")


def seed_worker(worker_id):
    """
    Seed function for DataLoader workers to ensure reproducible data loading.
    
    This should be passed to DataLoader via the worker_init_fn parameter:
        DataLoader(..., worker_init_fn=seed_worker)
    
    Args:
        worker_id (int): Worker ID (automatically passed by DataLoader)
    """
    # Each worker gets a unique but deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    """
    Create a PyTorch Generator with a specific seed for DataLoader.
    
    This ensures reproducible data shuffling when using shuffle=True:
        DataLoader(..., shuffle=True, generator=get_generator(seed))
    
    Args:
        seed (int): Random seed
        
    Returns:
        torch.Generator: Seeded generator
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g

