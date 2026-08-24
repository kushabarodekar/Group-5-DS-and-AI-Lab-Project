#!/usr/bin/env python3
"""Safe CUDA probe — never calls get_device_name() unless CUDA is available."""
import torch

available = torch.cuda.is_available()
print("cuda available:", available)
if available:
    print("device count:", torch.cuda.device_count())
    print("device 0:", torch.cuda.get_device_name(0))
else:
    print("No GPU on this machine — the app will run on CPU (expected on CPU-only Lightning studios).")
