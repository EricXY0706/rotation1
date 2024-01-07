import torch
import random
import os
import numpy as np
import pandas as pd
from colorama import Fore
from matplotlib import pyplot as plt
from sklearn import manifold

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def data_loader():
    NP_table_human = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_>20_85+merge.csv')
    NP_table_others = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_others_family.csv')
    NP_table_others = NP_table_others[NP_table_others['human'] == 'others']
    NP_human_list = NP_table_human['Sequence'].tolist()
    NP_human_family_list = NP_table_human['Family'].tolist()
    NP_others_list = NP_table_others['Sequence'].tolist()
    NP_tag = ['human'] * len(NP_human_list)
    NP_tag.extend(['others'] * len(NP_others_list))
    return NP_human_list, NP_others_list, NP_tag, NP_human_family_list
def metrics(TP, FP, TN, FN):
    ACC = round((TP + TN) / (TP + FN + TN + FP), 4)
    Precision = round(TP / (TP + FP), 4)
    Recall = round(TP / (TP + FN), 4)
    Sensitivity = round(TP / (TP + FN), 4)
    Specificity = round(TN / (FP + TN), 4)
    MCC = round((TP * TN - FP * FN) / pow((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 0.5), 4)
    F1_score = round(2 * Precision * Recall / (Precision + Recall), 4)
    print(f'Accuracy: {Fore.RED}{ACC}{Fore.RESET}')
    print(f'Precision: {Fore.RED}{Precision}{Fore.RESET}')
    print(f'Recall: {Fore.RED}{Recall}{Fore.RESET}')
    print(f'Sensitivity: {Fore.RED}{Sensitivity}{Fore.RESET}')
    print(f'Specificity: {Fore.RED}{Specificity}{Fore.RESET}')
    print(f'Matthews coefficient: {Fore.RED}{MCC}{Fore.RESET}')
    print(f'F1-score: {Fore.RED}{F1_score}{Fore.RESET}')
    # return ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score
def t_sne(x, y, file_name):
    x_max = torch.max(x, dim=1, keepdim=True).values
    x_min = torch.min(x, dim=1, keepdim=True).values
    x = (x - x_min) / (x_max - x_min)
    x = x.numpy()
    y = y.numpy()
    TSNE_transfer = manifold.TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, init='pca', random_state=42)
    seq_reps_tsne_emb = TSNE_transfer.fit_transform(x)
    plt.figure()
    seq_reps_tsne_emb_human_x, seq_reps_tsne_emb_human_y = [], []
    seq_reps_tsne_emb_others_x, seq_reps_tsne_emb_others_y = [], []
    for seq_rep_id in range(seq_reps_tsne_emb.shape[0]):
        tag = y[seq_rep_id, 0]
        if tag == 1:
            seq_reps_tsne_emb_human_x.append(x[seq_rep_id, 0])
            seq_reps_tsne_emb_human_y.append(x[seq_rep_id, 1])
        else:
            seq_reps_tsne_emb_others_x.append(x[seq_rep_id, 0])
            seq_reps_tsne_emb_others_y.append(x[seq_rep_id, 1])
    plt.scatter(seq_reps_tsne_emb_human_x, seq_reps_tsne_emb_human_y, s=10, color='rosybrown', alpha=0.7, label='Human')
    plt.scatter(seq_reps_tsne_emb_others_x, seq_reps_tsne_emb_others_y, s=10, color='royalblue', alpha=0.7, label='Others')
    # plt.title('t-SNE embedding of peptides in human or others after augmentation')
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.legend()
    plt.savefig(rf'/home/xuyi/rotation1/binary_model_results/ablation_tSNE/{file_name}.tiff', dpi=300, bbox_inches='tight')
    plt.show()