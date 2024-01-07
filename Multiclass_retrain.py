import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import random
import time
from colorama import Fore
from utils import set_seed
import numpy as np
from einops import rearrange

class multiple_classification_model(nn.Module):
    def __init__(self, max_seq_length, emb_length, proj_length, kernel_size, kernels, heads, num_classes):
        super(multiple_classification_model, self).__init__()
        self.kernels = kernels
        self.heads = heads
        self.projection = nn.Linear(in_features=emb_length, out_features=proj_length, bias=True)
        self.conv1 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        # self.conv3 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        # self.conv4 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        # self.conv5 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        # self.conv6 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        # self.conv7 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        # self.conv8 = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='same')
        self.attn = nn.Linear(in_features=int(proj_length / heads), out_features=max_seq_length, bias=True)
        self.classification = nn.Sequential(nn.Linear(in_features=proj_length, out_features=256, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=256, out_features=128, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=128, out_features=64, bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(in_features=64, out_features=num_classes, bias=True))
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
        # representations_conv += F.relu(self.conv3(representations_proj))
        # representations_conv += F.relu(self.conv4(representations_proj))
        representations_conv /= self.kernels # [b, p, s]
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
        return representations_prob, torch.argmax(representations_prob, dim=1)
# class multiple_classification_model(nn.Module):
    # def __init__(self, input_dim, num_classes=35):
    #     super(multiple_classification_model, self).__init__()
    #     self.classification = nn.Sequential(nn.Linear(in_features=input_dim, out_features=256, bias=True),
    #                                         nn.ReLU(),
    #                                         nn.Linear(in_features=256, out_features=128, bias=True),
    #                                         nn.ReLU(),
    #                                         nn.Linear(in_features=128, out_features=64, bias=True),
    #                                         nn.ReLU(),
    #                                         nn.Linear(in_features=64, out_features=num_classes, bias=True))
    # def forward(self, x):
    #     out = F.softmax(self.classification(x), dim=1)
    #     return out, torch.argmax(out, dim=1)
class MakeDataset(Dataset):
    def __init__(self, x, y, l):
        self.x = x
        self.y = y
        self.l = l
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx], 'l': self.l[idx]}
def model(parameters_dict):
    positive_classes = parameters_dict['positive_classes']
    positive_num = parameters_dict['positive_num']
    negtive_dataset = parameters_dict['negtive_dataset']
    binary_model_name = parameters_dict['binary_model_name']
    train_test_ratio = parameters_dict['train_test_ratio']
    lr = parameters_dict['learning_rate']
    epochs = parameters_dict['epochs']
    batch_size = parameters_dict['batch_size']
    proj_length = parameters_dict['projection_length']
    kernel_size = parameters_dict['kernel_size']
    n_kernels = parameters_dict['n_kernels']
    n_heads = parameters_dict['n_heads']
    NP_human_family_index = torch.load(rf'/home/xuyi/rotation1/binary_embeddings/y_{binary_model_name}_{str(positive_classes-1)}*{str(positive_num)}+merge_{negtive_dataset}.pt').to(device)
    ######################## Method 1: ESM -> FCNN
    # binary_out_human = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/tokens/tokens_esm2_t33_650M_UR50D.pt').to(device)
    # binary_out_human = binary_out_human[:NP_human_family_index.shape[0], :, :]
    # seq_length = torch.load(r'/home/xuyi/rotation1/ESM_embeddings/seq_length/seq_length_esm2_t33_650M_UR50D.pt').to(device)
    # seq_length = seq_length[:NP_human_family_index.shape[0]]
    # for i, l in enumerate(seq_length):
    #     l = int(l.item())
    #     binary_out_human[i, 0, :] = 0
    #     binary_out_human[i, l+1:, :] = 0
    # binary_out_human = torch.sum(binary_out_human, dim=1)
    # for i, l in enumerate(seq_length):
    #     l = int(l.item())
    #     binary_out_human[i, :] = binary_out_human[i, :] / l
    ##############################################
    ######################## Method 2: ESM -> CNN & Attention -> FCNN
    binary_out_human = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/tokens/tokens_esm2_t33_650M_UR50D_{str(positive_classes-1)}*{str(positive_num)}+merge_{negtive_dataset}.pt').to(device)
    binary_out_human = binary_out_human[:NP_human_family_index.shape[0], :, :]
    seq_length = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/seq_length/seq_length_esm2_t33_650M_UR50D_{str(positive_classes-1)}*{str(positive_num)}+merge_{negtive_dataset}.pt').to(device)
    seq_length = seq_length[:NP_human_family_index.shape[0]]
    ##############################################
    # Train & test splitting in every class
    max_seq_length, emb_length = binary_out_human.shape[1], binary_out_human.shape[2]
    reps_train, reps_test = torch.zeros(1, max_seq_length, emb_length).to(device), torch.zeros(1, max_seq_length, emb_length).to(device)
    target_train, target_test = torch.zeros(1, positive_classes).to(device), torch.zeros(1, positive_classes).to(device)
    length_train, length_test = torch.zeros(1).to(device), torch.zeros(1).to(device)
    for class_id in range(positive_classes):
        train_set_index = torch.tensor(random.sample(range(0, positive_num), int(positive_num * train_test_ratio))).to(device)
        test_set_index = torch.tensor(list(set(list(range(0, positive_num))) - set(train_set_index.cpu().numpy().tolist()))).to(device)
        reps_train = torch.cat((reps_train, binary_out_human[class_id * positive_num : (class_id + 1) * positive_num, :, :].index_select(0, train_set_index)), dim=0)
        target_train = torch.cat((target_train, NP_human_family_index[class_id * positive_num : (class_id + 1) * positive_num, :].index_select(0, train_set_index)), dim=0)
        length_train = torch.cat((length_train, seq_length[class_id * positive_num : (class_id + 1) * positive_num].index_select(0, train_set_index)), dim=0)
        reps_test = torch.cat((reps_test, binary_out_human[class_id * positive_num : (class_id + 1) * positive_num, :, :].index_select(0, test_set_index)), dim=0)
        target_test = torch.cat((target_test, NP_human_family_index[class_id * positive_num : (class_id + 1) * positive_num, :].index_select(0, test_set_index)), dim=0)
        length_test = torch.cat((length_test, seq_length[class_id * positive_num : (class_id + 1) * positive_num].index_select(0, test_set_index)), dim=0)
    reps_train, reps_test = reps_train[1:, :, :], reps_test[1:, :, :]
    target_train, target_test = target_train[1:, :], target_test[1:, :]
    length_train, length_test = length_train[1:], length_test[1:]
    # Z-score
    reps_train = (reps_train - reps_train.mean(dim=1, keepdim=True)) / reps_train.std(dim=1, keepdim=True)
    reps_test = (reps_test - reps_test.mean(dim=1, keepdim=True)) / reps_test.std(dim=1, keepdim=True)
    # Training
    training_set = MakeDataset(reps_train, target_train, length_train)
    print(f'Now begins training...')
    multiple_model = multiple_classification_model(max_seq_length, emb_length, proj_length, kernel_size, n_kernels, n_heads, positive_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(multiple_model.parameters(), lr=lr, momentum=0, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False)
    multiple_model.train()
    for epoch in range(epochs):
        print(f'epoch {epoch + 1}...', end='')
        start_time_epoch = time.time()
        dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        total_loss = 0
        for batch in dataloader:
            x_train, y_train, l_train = batch['x'], batch['y'], batch['l']
            y_prob, _ = multiple_model(x_train, l_train, device)
            loss = criterion(y_prob, y_train)
            total_loss = total_loss + loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step(total_loss)
        avg_loss = round(total_loss.item() / reps_train.shape[0], 4)
        time_elapsed_epoch = round((time.time() - start_time_epoch) / 60, 2)
        print(f'average loss is {Fore.RED}{avg_loss}{Fore.RESET}, time elapsed: {time_elapsed_epoch} min...')
    print('Training ends, and now begins testing...')
    multiple_model.eval()
    _, y_pre = multiple_model(reps_test, length_test, device)
    y_true = torch.argmax(target_test, dim=1)
    micro_f1 = round(f1_score(y_true.cpu(), y_pre.cpu(), average='micro', zero_division=0.0), 4)
    macro_f1 = round(f1_score(y_true.cpu(), y_pre.cpu(), average='macro', zero_division=0.0), 4)
    acc = round(accuracy_score(y_true.cpu(), y_pre.cpu()), 4)
    _, _, f1, _ = precision_recall_fscore_support(y_true.cpu(), y_pre.cpu(), average=None, zero_division=0.0)
    non_zero_f1 = np.round(np.mean(np.array(f1)[np.array(f1) != 0]), 4)
    print(f'Micro F1 score: {Fore.RED}{micro_f1}{Fore.RESET}')
    print(f'Macro F1 score: {Fore.RED}{macro_f1}{Fore.RESET}')
    print(f'Non-zero F1 score: {Fore.RED}{non_zero_f1}{Fore.RESET}')
    print(f'Accuracy: {Fore.RED}{acc}{Fore.RESET}')
    print(f1)
    return micro_f1, macro_f1, non_zero_f1
if __name__ == '__main__':
    # mF1, MF1, non = [], [], []
    # for seed in range(1, 100):
    #     set_seed(seed=seed)
    #     print(seed)
    #     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # params = {
        #     "positive_classes": 20,
		# 	"positive_num": 90,
   		# 	"negtive_dataset": 'allneg',
        #     "binary_model_name": 'e100b16_p512k3n1h4_BCESGD',
        #     "train_test_ratio": 0.8,
        #     "learning_rate": 1e-1,
        #     "epochs": 120,
        #     "batch_size": 125,
        #     "projection_length": 512,
        #     "kernel_size": 3,
        #     "n_kernels": 1,
        #     "n_heads": 1,
        # }
    #     micro_f1, macro_f1, non_zero_f1 = model(params)
    #     mF1.append(micro_f1)
    #     MF1.append(macro_f1)
    #     non.append(non_zero_f1)
    #     pass
    # print(np.max(np.array(mF1)), np.argmax(np.array(mF1)))
    # print(np.max(np.array(MF1)), np.argmax(np.array(MF1)))
    # print(np.max(np.array(non)), np.argmax(np.array(non)))
    set_seed(seed=42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = {
        	"positive_classes": 21,
			"positive_num": 85,
   			"negtive_dataset": 'allneg',
            "binary_model_name": 'e100b16_p512k3n2h4_BCESGD',
            "train_test_ratio": 0.8,
            "learning_rate": 1e-1,
            "epochs": 120,
            "batch_size": 125,
            "projection_length": 512,
            "kernel_size": 3,
            "n_kernels": 2,
            "n_heads": 4,
    }
    micro_f1, macro_f1, non_zero_f1 = model(params)