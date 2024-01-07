import pandas as pd
from colorama import Fore as F
from matplotlib import pyplot as plt
# from matplotlib_venn import venn2
from collections import Counter
import esm
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
print(alphabet)
PLM_seqs_train, PLM_seqs_test = [], []
PLM_train = pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\train.csv')
for row in PLM_train.iterrows():
    if row[1].label == 1:
        PLM_seqs_train.append(row[1].seq)
PLM_test = pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\test.csv')
for row in PLM_test.iterrows():
    if row[1].label == 1:
        PLM_seqs_test.append(row[1].seq)
PLM_seqs_train_1, PLM_seqs_test_1 = list(set(PLM_seqs_train)), list(set(PLM_seqs_test))
print(f'Before dereplication, Number of seqs in PLM-train: {F.RED}{len(PLM_seqs_train)}{F.RESET}, PLM-test: {F.RED}{len(PLM_seqs_test)}{F.RESET}')
print(f'After dereplication, Number of seqs in PLM-train: {F.RED}{len(PLM_seqs_train_1)}{F.RESET}, PLM-test: {F.RED}{len(PLM_seqs_test_1)}{F.RESET}')
result = Counter(PLM_seqs_train)
for key in result.keys():
    if result[key] > 1:
        print(f'{F.RED}{key}{F.RESET} is present for {F.RED}{result[key]}{F.RESET} times in PLM-train')
redundant_seqs = []
for seq in PLM_seqs_train:
    if seq in PLM_seqs_test:
        redundant_seqs.append(seq)
redundant_seqs = list(set(redundant_seqs))
for s in redundant_seqs:
    print(f'Sequence {F.RED}{s}{F.RESET} is redundant between training set & testing set, and it appears {F.RED}{result[s]}{F.RESET} times in training set')
subsets = [set(PLM_seqs_train), set(PLM_seqs_test)]
labels = ('PLM_train', 'PLM_test')
plt.figure()
plt.title('PLM-train & PLM-test')
# g = venn2(subsets=subsets, set_labels=labels)
plt.show()

my_seqs = []
f = open(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_dereplication_5-100_CD-HIT.txt')
lines = f.readlines()
for line in lines:
    if line[0] != '>':
        my_seqs.append(line[:-1])
print(f'Number of seqs in Mine: {F.RED}{len(my_seqs)}{F.RESET}')

subsets = [set(PLM_seqs_train+PLM_seqs_test), set(my_seqs)]
labels = ('PLM', 'Mine')
plt.figure()
plt.title('PLM & Mine')
# g = venn2(subsets=subsets, set_labels=labels)
plt.show()