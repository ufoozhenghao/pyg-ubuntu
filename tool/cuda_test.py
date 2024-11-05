import torch

print("PyTorch version:", torch.__version__)

cuda_available = torch.cuda.is_available()
print("CUDA available:", cuda_available)

if cuda_available:
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available.")

import matplotlib
print(matplotlib.__version__)
import numpy as np
print(np.__version__)