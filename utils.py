import torch
import random
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import esm
from tqdm import *
from colorama import Fore

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def data_loader():
    NP_human_file = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_0109_20*90.csv')
    NP_others_file = pd.read_csv(r'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_others_family.csv')
    NP_others_file = NP_others_file[NP_others_file['human'] == 'others']
    NP_human_list = NP_human_file['Sequence'].tolist()
    NP_human_family_list = NP_human_file['Family'].tolist()
    NP_others_list = NP_others_file['Sequence'].tolist()
    NP_tag = ['human'] * len(NP_human_list) + ['others'] * len(NP_others_list)
    return NP_human_list, NP_others_list, NP_tag, NP_human_family_list
def metrics(TP, FP, TN, FN):
    Precision = round(TP / (TP + FP), 4)
    Recall = round(TP / (TP + FN), 4)
    F1 = round(2 * Precision * Recall / (Precision + Recall), 4)
    Sensitivity = round(TP / (TP + FN), 4)
    Specificity = round(TN / (FP + TN), 4)
    ACC = round((TP + TN) / (TP + FN + TN + FP), 4)
    MCC = round((TP * TN - FP * FN) / pow((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 0.5), 4)
    print(f'ACC: {Fore.RED}{ACC}{Fore.RESET}.')
    print(f'Precision: {Fore.RED}{Precision}{Fore.RESET}.')
    print(f'Recall: {Fore.RED}{Recall}{Fore.RESET}.')
    print(f'Sensitivity: {Fore.RED}{Sensitivity}{Fore.RESET}.')
    print(f'Specificity: {Fore.RED}{Specificity}{Fore.RESET}.')
    print(f'MCC: {Fore.RED}{MCC}{Fore.RESET}.')
    print(f'F1: {Fore.RED}{F1}{Fore.RESET}.')