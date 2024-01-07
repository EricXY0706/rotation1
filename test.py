import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from einops import rearrange, repeat, reduce
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import wandb
import random
import torch.nn as nn

# start a new wandb run to track this script
a = torch.rand(1, 5, 2, 3)
c = torch.randint(1, 5, (1, 2, 2, 3))
b = nn.Linear(7, 8)

print(torch.mean(a, dim=1).shape)