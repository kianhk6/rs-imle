import torch
import os
from dit import DiT_S_2  # Import DiT-S/2 model

def remove_module_prefix(state_dict, prefix="module.decoder."):
    """Removes a prefix from checkpoint keys if it exists"""
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k[len(prefix):] if k.startswith(prefix) else k
        new_state_dict[new_key] = v
    return new_state_dict


# Path to the saved model checkpoint
checkpoint_path = "/home/kha98/rs-imle/flowers-results/ffhq/train/model_epoch_0.pt"

# Check if the checkpoint exists
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    exit()

# Load the checkpoint
# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Remove 'module.decoder.' prefix from keys if necessary
checkpoint["model_state_dict"] = remove_module_prefix(checkpoint["model_state_dict"])

# Initialize model
model = DiT_S_2()

# Load the modified state_dict
model.load_state_dict(checkpoint["model_state_dict"], strict=False)
print("✅ Model state_dict loaded successfully!")


# Print available keys in the checkpoint
print("Checkpoint Keys:", checkpoint.keys())

# Ensure required keys are present
expected_keys = ["epoch", "model_state_dict", "ema_model_state_dict", "optimizer_state_dict"]
missing_keys = [key for key in expected_keys if key not in checkpoint]
if missing_keys:
    print(f"Warning: Missing keys in checkpoint: {missing_keys}")

model = DiT_S_2()


# Load model weights
model.load_state_dict(checkpoint["model_state_dict"])
print("✅ Model state_dict loaded successfully!")



# Print optimizer state
if "optimizer_state_dict" in checkpoint:
    print("✅ Optimizer state_dict is available.")
else:
    print("⚠️ Optimizer state_dict is missing.")

# Print saved epoch
print(f"✅ Model was saved at epoch: {checkpoint['epoch']}")

# Perform a forward pass with a dummy input
dummy_input = torch.randn(1, 4, 32, 32)  # Modify dimensions if needed
try:
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        output = model(dummy_input)  # Forward pass
    print("✅ Model forward pass successful! Output shape:", output.shape)
except Exception as e:
    print("❌ Error during forward pass:", str(e))
