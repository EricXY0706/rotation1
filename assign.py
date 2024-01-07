import pandas as pd
from matplotlib import pyplot as plt
import matplotlib_venn
from collections import Counter
seq = pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_all_1.csv')['Sequence']
fam = pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_all_1.csv')['Family']
receptor = pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_all_1.csv')['Receptor']
org = pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_all_1.csv')['Organism']
df = pd.DataFrame()
df['Sequence'] = seq
df['Family'] = fam
df['Receptor'] = receptor
df['Organism'] = org
df_new = df.drop_duplicates()
df_new.set_index('Sequence', inplace=True)

CD_hit_file = open(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_7-100_CD-HIT.txt')
CD_hit_lines = CD_hit_file.readlines()
result_seq, result_length, result_fam, result_recep, result_org = [], [], [], [], []
for line in CD_hit_lines:
    if line[0] != '>':
        result_seq.append(line[:-1])
        result_length.append(len(line[:-1]))
        result_fam.append(df_new.loc[line[:-1], 'Family'])
        result_recep.append(df_new.loc[line[:-1], 'Receptor'])
        result_org.append(df_new.loc[line[:-1], 'Organism'])
result_df = pd.DataFrame()
result_df['Sequence'] = pd.Series(result_seq)
result_df['Length'] = pd.Series(result_length)
result_df['Family'] = pd.Series(result_fam)
result_df['Receptor'] = pd.Series(result_recep)
result_df['Organism'] = pd.Series(result_org)
result_df.to_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_7-100_CD-HIT_Family.csv', index=False)
fig, ax = plt.subplots()
v1 = ax.violinplot(result_length, vert=True, showmeans=True, showmedians=True)
labels = ['NeuroPep_CD_HIT']
x = range(len(labels) + 1)
plt.xticks(x[1:], labels)
plt.ylabel('Length', fontsize=15)
plt.xlabel('Peptide', fontsize=15)
# plt.savefig(r'D:\Xuyi\UESTC\2 Scientific Research\7 ACVPpred\0701\seq_length.tiff', dpi=300, bbox_inches='tight')
# plt.savefig(r'D:\Xuyi\UESTC\2 Scientific Research\7 ACVPpred\0701\seq_length.svg', bbox_inches='tight')
plt.show()