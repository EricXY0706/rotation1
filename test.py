import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from einops import rearrange, repeat, reduce
import random
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_curve, auc, confusion_matrix
from utils import set_seed, data_loader

# set_seed(42)
# file = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family.csv').loc[:, ['Sequence', 'Family', 'Length']]
# fam = file['Family'].tolist()
# fam_stats = Counter(fam)
# print(fam_stats)
# for fam in fam_stats.keys():
# 	if fam_stats[fam] < 10:
# 		file = file[file['Family'] != fam]
# # for fam in fam_stats.keys():
# # 	if 10 < fam_stats[fam] <= 20:
# # 		file.loc[file['Family'] == fam, 'Family'] = 'MergeOthers'
# fam = file['Family'].tolist()
# fam_stats = Counter(fam)
# print(fam_stats)
# file = file.sort_values(by='Family')
# file.to_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_0313.csv')

# for fam in fam_stats.keys():
#     if fam_stats[fam] >= 20:
#         file = file[file['Family'] != fam]
# fam_new = file['Family'].tolist()
# fam_stats_new = Counter(fam_new)
# print(fam_stats_new)
# results = []
# for fam in fam_stats_new:
#     if fam_stats_new[fam] == 19:
#         items = []
#         file_fam = file[file['Family'] == fam].values.tolist()
#         indices = random.sample(range(fam_stats_new[fam]), 9)
#         for index in indices:
#             items.append(file_fam[index])		
#         results.extend(items)
#     elif 8 <= fam_stats_new[fam] < 19:
#         items = []
#         file_fam = file[file['Family'] == fam].values.tolist()
#         indices = random.sample(range(fam_stats_new[fam]), 8)
#         for index in indices:
#             items.append(file_fam[index])		
#         results.extend(items)
#     elif 3 <= fam_stats_new[fam] <= 7:
#         items = []
#         file_fam = file[file['Family'] == fam].values.tolist()
#         indices = random.sample(range(fam_stats_new[fam]), 3)
#         for index in indices:
#             items.append(file_fam[index])		
#         results.extend(items)
#     else:
#         items = file[file['Family'] == fam].values.tolist()
#         results.extend(items)
# df = pd.DataFrame(results)
# df.columns = ['Sequence', 'Family', 'Length']
# df['Family'] = pd.Series(['MergeOthers'] * df.shape[0])
# df.to_csv(r'/home/xuyi/rotation1/datasets/merge.csv', index=False)
file = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_0109.csv')
fams = file['Family'].tolist()
print(Counter(fams))
result = list(Counter(fams).keys())
result.sort()
print(len(result))