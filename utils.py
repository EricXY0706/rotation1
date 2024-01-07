import torch
import random
import os
import numpy as np
import pandas as pd
from colorama import Fore

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def data_loader():
    NP_table_human = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_60.csv')
    NP_table_others = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_others_family.csv')
    NP_human_list = NP_table_human['Sequence'].tolist()
    NP_others_list = NP_table_others['Sequence'].tolist()
    NP_tag = ['human'] * len(NP_human_list)
    NP_tag.extend(['others'] * len(NP_others_list))
    return NP_human_list, NP_others_list, NP_tag
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
    return ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score