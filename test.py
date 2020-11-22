import numpy as np
import torch

index = torch.tensor([[1], [0]])
map = {0: [1, 2, 3], 1: [4, 5, 6]}
res = torch.gather(map, 1, index)
print(res)
