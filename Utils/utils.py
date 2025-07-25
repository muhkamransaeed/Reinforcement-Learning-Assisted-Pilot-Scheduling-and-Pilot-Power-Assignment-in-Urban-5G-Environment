import numpy as np
import json
import os

def normalize(value, min_val, max_val):
    """Min-max normalize a value or numpy array to [0,1]."""
    return (value - min_val) / (max_val - min_val + 1e-8)

def denormalize(norm_value, min_val, max_val):
    """Convert normalized value back to original scale."""
    return norm_value * (max_val - min_val) + min_val

def save_json(data, filepath):
    """Save dictionary or list data to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def load_json(filepath):
    """Load JSON file to dictionary or list."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    with open(filepath, 'r') as f:
        return json.load(f)

def moving_average(data, window_size=10):
    """Compute moving average with given window size."""
    if len(data) < window_size:
        return np.array(data)
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def clip_array(arr, min_val, max_val):
    """Clip numpy array or list elements to specified range."""
    return np.clip(arr, min_val, max_val)

def set_seed(seed):
    """Set seed for reproducibility."""
    import random
    import torch

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
