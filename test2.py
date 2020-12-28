import numpy as np
import torch

arr = np.array([0.1, 0.2, 0.1, 0.8, 0.2, 0.3, 0.2, 0.2, 0.2, 0.2, 0.1, 0.2])
arg = arr.argsort(kind='stable')
arg = arg[-4:].tolist()
arg.reverse()
print(arg)

# arr = torch.FloatTensor(
#     [0.1, 0.2, 0.1, 0.8, 0.2, 0.3, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1])
# _, idx = arr.topk(4, largest=True, dim=0)
# _, idx = arr.sort(descending=True)
# print(idx[:4].numpy())
