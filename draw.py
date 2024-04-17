import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
def mulF1():
    precision = [0.92857143, 1.,         0.52380952, 0.63636364, 0.84210526, 0.93333333,
 0.70588235, 0.88888889, 0.85,       0.45833333, 0.47368421, 1.,
 0.5 ,       0.83333333, 0.71428571, 0.875,      0.77777778, 0.76470588,
 0.88888889, 0.85714286]
    recall = [0.72222222, 0.88888889, 0.61111111, 0.77777778, 0.88888889, 0.77777778,
 0.66666667, 0.88888889, 0.94444444, 0.61111111, 0.5,        0.88888889,
 0.44444444, 0.55555556, 0.83333333, 0.77777778, 0.77777778, 0.72222222,
 0.88888889, 1.        ]
    f1 = [0.8125,     0.94117647, 0.56410256, 0.7,        0.86486486, 0.84848485,
 0.68571429, 0.88888889, 0.89473684, 0.52380952, 0.48648649, 0.94117647,
 0.47058824, 0.66666667, 0.76923077, 0.82352941, 0.77777778, 0.74285714,
 0.88888889, 0.92307692]
    fam = ['Bombesin/neuromedin-B/ranatensin', 'Calcitonin', 'Chromogranin/secretogranin', 'FMRFamide related peptide', 'Galanin', 'Gastrin/cholecystokinin', 'Glucagon', 'GnRH', 'Insulin', 'Others', 'NPY', 'Natriuretic peptide', 'Opioid', 'POMC', 'RFamide neuropeptide', 'Sauvagine/corticotropin-releasing factor/urotensin I', 'Serpin', 'Somatostatin', 'Tachykinin', 'Vasopressin/oxytocin']
    df = pd.DataFrame()
    df['fam'] = pd.Series(fam)
    df['Score'] = pd.Series(f1)
    df.sort_values(by='Score', ascending=False, inplace=True)
    plt.figure()
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.bar(df['fam'].tolist(), df['Score'].tolist(), alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9])
    for x, num in enumerate(df['Score'].tolist()):
        plt.text(x, num+0.008, s=str(round(num, 2)), ha='center', fontsize=6)
    plt.ylabel('F1 score', fontsize=12)
    plt.savefig(rf'/home/xuyi/rotation1/multiple_model_results/f1.pdf', dpi=300, bbox_inches='tight')
def ori_num():
    file = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family.csv')
    fam = file['Family'].tolist()
    fam_stats = Counter(fam)
    fam_stats = sorted(fam_stats.items(), key=lambda x: x[1], reverse=True)
    x, y = [], []
    for fam_pair in fam_stats:
        x.append(fam_pair[0])
        y.append(fam_pair[1])
    plt.figure(figsize=(20, 10))
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    color = 'red'
    plt.bar(x, y, width=0.9, color=color, alpha=0.4)
    plt.xticks(rotation=45, ha='right', fontsize=14)
    plt.ylabel('Number', fontsize=12)
    plt.savefig(rf'/home/xuyi/rotation1/datasets/original_family_numbers_{color}.pdf', dpi=300, bbox_inches='tight')
def mulConfusion():
    confusions = np.load(r'/home/xuyi/rotation1/multiple_model_results/confusions.npy')
    fam = ['Bombesin/neuromedin-B/ranatensin', 'Calcitonin', 'Chromogranin/secretogranin', 'FMRFamide related peptide', 'Galanin', 'Gastrin/cholecystokinin', 'Glucagon', 'GnRH', 'Insulin', 'Others', 'NPY', 'Natriuretic peptide', 'Opioid', 'POMC', 'RFamide neuropeptide', 'Sauvagine/corticotropin-releasing factor/urotensin I', 'Serpin', 'Somatostatin', 'Tachykinin', 'Vasopressin/oxytocin']
    plt.imshow(confusions, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(fam)), fam, rotation=45, ha='right')
    plt.yticks(np.arange(len(fam)), fam)
    for i in range(len(fam)):
        for j in range(len(fam)):
            if round(confusions[i, j], 2) > 0.:
                if i == j:
                    plt.text(j, i, str(round(confusions[i, j], 2)), ha='center', va='center', color='white', fontsize=5)
                else:
                    plt.text(j, i, str(round(confusions[i, j], 2)), ha='center', va='center', color='black', fontsize=5)
    plt.savefig(r'/home/xuyi/rotation1/multiple_model_results/confusions.pdf', dpi=300, bbox_inches='tight')

# mulF1()
# ori_num()
mulConfusion()