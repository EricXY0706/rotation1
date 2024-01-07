import pandas as pd
seqs = list(set(pd.read_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_all.csv')['Sequence'].tolist()))
seqs_, lengths = [], []
for seq in seqs:
    if 7 <= len(seq) <= 100:
        seqs_.append(seq)
        lengths.append(len(seq))
df = pd.DataFrame()
df['Sequence'] = pd.Series(seqs_)
df['Length'] = pd.Series(lengths)
df.to_csv(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_7-100.csv', index=False)
with open(r'D:\Xuyi\THU\1_Science\rotation1\datasets\NeuroPepV2_neuropeptide_7-100.txt', 'w') as fileobj:
    for i in range(len(seqs_)):
        fileobj.write('>NP'+str(i+1)+'\n')
        fileobj.write(seqs_[i]+'\n')