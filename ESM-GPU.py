import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from einops import rearrange
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import *
import random
from colorama import Fore
import os
import wandb
import time
from utils import set_seed, data_loader, metrics

class binary_classification_model(nn.Module):
    def __init__(self, max_seq_length, emb_length, proj_length, kernel_size, kernels, heads):
        super(binary_classification_model, self).__init__()
        self.kernels = kernels
        self.heads = heads
        self.projection = nn.Linear(in_features=emb_length, out_features=proj_length, bias=True)
        self.conv1 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv3 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv4 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv5 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv6 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.attn = nn.Linear(in_features=proj_length / heads, out_features=max_seq_length, bias=True)
        self.classification = nn.Sequential(nn.Linear(in_features=proj_length / heads, out_features=int(proj_length / heads / 2), bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=int(proj_length / heads / 2), out_features=2, bias=True))
    def forward(self, x, seq_length):
        # x: [batch_size, max_seq_length, emb_length] -> [b, s, e]
        representations_proj = self.projection(x) # [b, s, e] -> [b, s, p]
        representations_proj = rearrange(representations_proj, 'b s p -> b p s')
        for i, l in enumerate(seq_length):
            l = l.item()
            representations_proj[i, :, 0] = 0
            representations_proj[i, :, l+1:] = 0
        representations_conv = torch.zeros_like(representations_proj)
        for k in range(self.kernels):
            representations_conv = representations_conv + F.relu(globals()['self.conv'+str(k+1)](representations_proj))
        representations_conv /= self.kernels # [b, p, s]
        representations_conv = rearrange(representations_conv, 'b (l h) s -> b h s l', h=self.heads)
        attn_score = self.attn(representations_conv) # [b, h, s, s]
        for i, l in enumerate(seq_length):
            l = l.item()
            attn_score[i, :, :, 0] = -1e9
            attn_score[i, :, :, l+1:] = -1e9
        attn_score = F.softmax(attn_score, dim=-1)
        representations_attn = torch.matmul(attn_score, representations_conv) # [b, h, s, l]
        representations_attn = torch.mean(representations_attn, dim=1) # [b, s, l]
        for i, l in enumerate(seq_length):
            representations_attn[i, 0, :] = 0
            representations_attn[i, l+1:, :] = 0
        representations_attn = torch.sum(representations_attn, dim=1) # [b, l]
        representations_prob = F.softmax(self.classification(representations_attn), dim=1) # [b, 2]
        return representations_prob, torch.max(representations_prob, dim=1).indices
class MakeDataset(Dataset):
    def __init__(self, x, y, l):
        self.x = x
        self.y = y
        self.l = l
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx], 'l': self.l[idx]}
def model(ESM_model_name, result_name, parameters_dict):
    set_seed(seed=42)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    training_test_ratio = parameters_dict['training_ratio']
    epochs = parameters_dict['epochs']
    kfolds = parameters_dict['kfolds']
    batch_size = parameters_dict['batch_size']
    proj_length = parameters_dict['projection_length']
    kernel_size = parameters_dict['kernel_size']
    n_kernels = parameters_dict['n_kernels']
    n_heads = parameters_dict['n_heads']
    lr = parameters_dict['learning_rate']
    token_representations = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/tokens/tokens_{ESM_model_name}.pt').to(device)
    sequence_length = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/seq_length/seq_length_{ESM_model_name}.pt').to(device)
    _, _, NP_tag = data_loader()
    total_size, max_seq_length, emb_length = token_representations.shape[0], token_representations.shape[1], token_representations.shape[2]
    target_representations = torch.zeros(total_size, 2).to(device)
    for i, tag in enumerate(NP_tag):
        if tag == 'human':
            target_representations[i, 0] = 1
        else:
            target_representations[i, 1] = 1
    training_index = torch.tensor(random.sample(range(0, total_size), int(total_size * training_test_ratio))).to(device)
    testing_index = torch.tensor(list(set(list(range(0, total_size))) - set(training_index))).to(device)
    reps_train, reps_test = token_representations.index_select(0, training_index), token_representations.index_select(0, testing_index)
    target_train, target_test = target_representations.index_select(0, training_index), target_representations.index_select(0, testing_index)
    length_train, length_test = sequence_length.index_select(0, training_index), sequence_length.index_select(0, testing_index)
    dataset = MakeDataset(reps_train, target_train, length_train)
    colors = ['#457479', '#432B3E', '#D64B3F', '#E5855F', '#EED5B9', '#853F29']
    results, AUCs = [], []
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    print('Now begins training...')
    plt.figure()
    if not os.path.exists(rf'/home/xuyi/rotation1/binary_model_results/CV/{result_name}'):
        os.mkdir(rf'/home/xuyi/rotation1/binary_model_results/CV/{result_name}')
    for k, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f'\nNow training No.{Fore.RED}{k + 1}{Fore.RESET} fold...')
        binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length,
                                                   proj_length=proj_length, kernel_size=kernel_size,
                                                   kernels=n_kernels, heads=n_heads).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.SGD(binary_model.parameters(), lr=lr, weight_decay=0)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        wandb.watch(binary_model, log='all')
        start_time_kfold = time.time()
        train_x, train_y, train_l = dataset[train_index]['x'], dataset[train_index]['y'], dataset[train_index]['l']
        val_x, val_y, val_l = dataset[val_index]['x'], dataset[val_index]['y'], dataset[val_index]['l']
        training_set = MakeDataset(train_x, train_y, train_l)
        # K-fold training
        binary_model.train()
        for epoch in range(epochs):
            print(f'\nepoch {Fore.RED}{epoch + 1}{Fore.RESET}...', end='')
            start_time_epoch = time.time()
            dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
            total_loss, minibatch_size = 0, 0
            for batch in dataloader:
                X, Y, L = batch['x'], batch['y'], batch['l']
                minibatch_size = L.shape[0]
                y_prob, y_pre = binary_model(X, L)
                loss = criterion(y_prob, Y)
                total_loss = total_loss + loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            scheduler.step(total_loss)
            avg_loss = round(total_loss.item() / minibatch_size, 4)
            wandb.log({"avg-loss for an epoch": avg_loss})
            time_elapsed_epoch = round((time.time() - start_time_epoch) / 60, 2)
            print(f'average loss is {Fore.RED}{avg_loss}{Fore.RESET}, time elapsed: {time_elapsed_epoch} min...')
        # K-fold testing
        binary_model.eval()
        y_prob, y_pre = binary_model(val_x, val_l)
        fpr, tpr, thresholds = roc_curve(val_y.numpy()[:, 0], y_prob.detach().numpy()[:, 0], pos_label=1)
        AUC = round(auc(fpr, tpr), 4)
        print(f'\nFold {k + 1}, training results on validation-set {k + 1}:')
        print(f'AUC: {Fore.RED}{AUC}{Fore.RESET}')
        confusions = confusion_matrix(torch.max(val_y, dim=1).indices.numpy(), y_pre.numpy())
        TP, FP, TN, FN = confusions[0][0], confusions[0][1], confusions[1][0], confusions[1][1]
        time_elapsed_kfold = round((time.time() - start_time_kfold) / 60, 2)
        ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score = metrics(TP, FP, TN, FN)
        print(f'Time elapsed: {Fore.RED}{time_elapsed_kfold}{Fore.RESET} min')
        # Record
        AUCs.append(AUC)
        plt.plot(fpr, tpr, color=colors[k], lw=2, label=f'fold{k}')
        results.append([k, AUC, ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score])
        torch.save(binary_model.state_dict(), rf'/home/xuyi/rotation1/binary_model_results/CV/{result_name}/k={k+1}.pth')
    print('Training ends, and now begins testing...')
    binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length,
                                               proj_length=proj_length, kernel_size=kernel_size, kernels=n_kernels, heads=n_heads).to(device)
    best_model_index = np.argmax(np.array(AUCs)) + 1
    best_model_path = rf'/home/xuyi/rotation1/binary_model_results/CV/{result_name}/k={best_model_index}.pth'
    binary_model.load_state_dict(torch.load(best_model_path))
    binary_model.eval()
    y_prob, y_pre = binary_model(reps_test, length_test)
    fpr, tpr, thresholds = roc_curve(target_test.numpy()[:, 0], y_prob.detach().numpy()[:, 0], pos_label=1)
    AUC = round(auc(fpr, tpr), 4)
    print(f'AUC: {Fore.RED}{AUC}{Fore.RESET}')
    confusions = confusion_matrix(torch.max(target_test, dim=1).indices.numpy(), y_pre.numpy())
    TP, FP, TN, FN = confusions[0][0], confusions[0][1], confusions[1][0], confusions[1][1]
    ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score = metrics(TP, FP, TN, FN)
    # Record
    plt.plot(fpr, tpr, color=colors[-1], lw=2, label=f'testing set')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.legend()
    plt.savefig(rf'/home/xuyi/rotation1/binary_model_results/ROC/{result_name}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    results.append(['testing set', AUC, ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score])
    result_df = pd.DataFrame(results)
    result_df.columns = ['Dataset', 'AUC', 'Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'MCC', 'F1']
    result_df.to_csv(rf'/home/xuyi/rotation1/binary_model_results/result/{result_name}.csv', index=False)
if __name__ == '__main__':
    # tsne_embedding(ESM_model_name='esm2_t33_650M_UR50D')
    wandb.init(project='binary_classification', entity='team-xy')
    params = {"training_ratio": 0.8,
              "epochs": 15,
              "kfolds": 5,
              "batch_size": 8,
              "projection_length": 512,
              "kernel_size": 3,
              "n_kernels": 2, # max n_kernels: 6
              "n_heads": 2,
              "learning_rate": 1e-2
              }
    project_name = 'e' + str(params['epochs']) + 'k' + str(params['kfolds']) + 'b' + str(params['batch_size']) + '_p' + \
                  str(params['projection_length']) + 'k' + str(params['kernel_size']) + 'n' + str(params['n_kernels']) + \
                  'h' + str(params['n_heads']) + '_BCEWithLogitsSGD'
    print(f'project name: {project_name}')
    model(ESM_model_name='esm2_t33_650M_UR50D', result_name=project_name, parameters_dict=params)
    pass