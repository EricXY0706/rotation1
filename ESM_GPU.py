import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import roc_curve, auc, confusion_matrix
from einops import rearrange
from matplotlib import pyplot as plt
from tqdm import *
import random
from colorama import Fore
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
        self.attn = nn.Linear(in_features=int(proj_length / heads), out_features=max_seq_length, bias=True)
        self.classification = nn.Sequential(nn.Linear(in_features=proj_length, out_features=256, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=256, out_features=128, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=128, out_features=64, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=64, out_features=32, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=32, out_features=16, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=16, out_features=8, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=8, out_features=4, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=4, out_features=2, bias=True))
    def forward(self, x, seq_length, device):
        # x: [batch_size, max_seq_length, emb_length] -> [b, s, e]
        representations_proj = self.projection(x) # [b, s, e] -> [b, s, p]
        representations_proj = rearrange(representations_proj, 'b s p -> b p s')
        for i, l in enumerate(seq_length):
            l = int(l.item())
            representations_proj[i, :, 0] = 0
            representations_proj[i, :, l+1:] = 0
        representations_conv = torch.zeros_like(representations_proj).to(device)
        representations_conv += F.relu(self.conv1(representations_proj))
        representations_conv += F.relu(self.conv2(representations_proj))
        # representations_conv /= self.kernels # [b, p, s]
        representations_conv = rearrange(representations_conv, 'b (l h) s -> b h s l', h=self.heads)
        attn_score = self.attn(representations_conv) # [b, h, s, s]
        for i, l in enumerate(seq_length):
            l = int(l.item())
            attn_score[i, :, :, 0] = -1e9
            attn_score[i, :, :, l+1:] = -1e9
        attn_score = F.softmax(attn_score, dim=-1)
        representations_attn = torch.matmul(attn_score, representations_conv) # [b, h, s, l]
        # representations_attn = torch.mean(representations_attn, dim=1) # [b, s, l]
        representations_attn = rearrange(representations_attn, 'b h s l -> b s (l h)') # [b, s, p]
        for i, l in enumerate(seq_length):
            l = int(l.item())
            representations_attn[i, 0, :] = 0
            representations_attn[i, l+1:, :] = 0
        representations_attn = torch.sum(representations_attn, dim=1) # [b, p]
        representations_prob = F.softmax(self.classification(representations_attn), dim=1) # [b, 2]
        return representations_prob, torch.max(representations_prob, dim=1).indices, representations_attn
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    positive_dataset = parameters_dict['positive_dataset']
    negative_dataset = parameters_dict['negative_dataset']
    train_test_ratio = parameters_dict['training_ratio']
    epochs = parameters_dict['epochs']
    batch_size = parameters_dict['batch_size']
    proj_length = parameters_dict['projection_length']
    kernel_size = parameters_dict['kernel_size']
    n_kernels = parameters_dict['n_kernels']
    n_heads = parameters_dict['n_heads']
    lr = parameters_dict['learning_rate']
    token_representations = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/tokens/tokens_{ESM_model_name}_{positive_dataset}_{negative_dataset}.pt').to(device)
    sequence_length = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/seq_length/seq_length_{ESM_model_name}_{positive_dataset}_{negative_dataset}.pt').to(device)
    _, _, NP_tag, _ = data_loader()
    total_size, max_seq_length, emb_length = token_representations.shape[0], token_representations.shape[1], token_representations.shape[2]
    target_representations = torch.zeros(total_size, 2).to(device)
    for i, tag in enumerate(NP_tag):
        if tag == 'human':
            target_representations[i, 0] = 1
        else:
            target_representations[i, 1] = 1
    training_index = torch.tensor(random.sample(range(0, total_size), int(total_size * train_test_ratio))).to(device)
    testing_index = torch.tensor(list(set(list(range(0, total_size))) - set(training_index.cpu().numpy().tolist()))).to(device)
    reps_train, reps_test = token_representations.index_select(0, training_index), token_representations.index_select(0, testing_index)
    target_train, target_test = target_representations.index_select(0, training_index), target_representations.index_select(0, testing_index)
    length_train, length_test = sequence_length.index_select(0, training_index), sequence_length.index_select(0, testing_index)
    dataset = MakeDataset(reps_train, target_train, length_train)
    print('Now begins training...')
    binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length,
                                                proj_length=proj_length, kernel_size=kernel_size,
                                                kernels=n_kernels, heads=n_heads).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(binary_model.parameters(), lr=lr, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    binary_model.train()
    for epoch in range(epochs):
        print(f'epoch {epoch + 1}...', end='')
        start_time_epoch = time.time()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        total_loss = 0
        for batch in dataloader:
            X, Y, L = batch['x'], batch['y'], batch['l']
            y_prob, y_pre, _ = binary_model(X, L, device)
            loss = criterion(y_prob, Y)
            total_loss = total_loss + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step(total_loss)
        avg_loss = round(total_loss.item() / reps_train.shape[0], 4)
        time_elapsed_epoch = round((time.time() - start_time_epoch) / 60, 2)
        print(f'average loss is {Fore.RED}{avg_loss}{Fore.RESET}, time elapsed: {time_elapsed_epoch} min...')
    print('Training ends, and now begins testing...')
    binary_model.eval()
    y_prob, y_pre, _ = binary_model(reps_test, length_test, device)
    fpr, tpr, _ = roc_curve(target_test.cpu().numpy()[:, 0], y_prob.cpu().detach().numpy()[:, 0], pos_label=1)
    AUC = round(auc(fpr, tpr), 4)
    print(f'AUC: {Fore.RED}{AUC}{Fore.RESET}')
    confusions = confusion_matrix(torch.max(target_test, dim=1).indices.cpu().numpy(), y_pre.cpu().numpy())
    TP, FP, TN, FN = confusions[1][1], confusions[0][1], confusions[0][0], confusions[1][0]
    metrics(TP, FP, TN, FN)
    # Record
    # torch.save(binary_model.state_dict(), rf'/home/xuyi/rotation1/binary_model_results/model/b_{result_name}_{positive_dataset}_{negative_dataset}.pth')
    plt.plot(fpr, tpr, color='red', lw=2, label=f'AUC={AUC}')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.legend()
    # plt.savefig(rf'/home/xuyi/rotation1/binary_model_results/ROC/{result_name}_{positive_dataset}_{negative_dataset}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    params = {"positive_dataset": '20*85+merge',
              "negative_dataset": 'allneg',
              "training_ratio": 0.8,
              "epochs": 100,
              "batch_size": 16,
              "projection_length": 512,
              "kernel_size": 3,
              "n_kernels": 2,
              "n_heads": 4,
              "learning_rate": 1e-2
              }
    project_name = 'e' + str(params['epochs']) +  'b' + str(params['batch_size']) + '_p' + \
                  str(params['projection_length']) + 'k' + str(params['kernel_size']) + 'n' + str(params['n_kernels']) + \
                  'h' + str(params['n_heads']) + '_BCESGD'
    print(f'project name: {project_name}')
    model(ESM_model_name='esm2_t33_650M_UR50D', result_name=project_name, parameters_dict=params)
    pass