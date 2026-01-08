import torch
import sys

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Python version: {sys.version}")
except Exception as e:
    print(f"Error checking torch versions: {e}")
