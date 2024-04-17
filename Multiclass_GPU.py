import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, recall_score, precision_score, confusion_matrix
import random
import time
from colorama import Fore
from utils import set_seed
import numpy as np
from tqdm import *

class multiple_classification_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(multiple_classification_model, self).__init__()
        self.classification = nn.Sequential(nn.Linear(in_features=input_dim, out_features=256, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=256, out_features=128, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=128, out_features=64, bias=True),
                                            nn.ReLU(),
                                            nn.Linear(in_features=64, out_features=num_classes, bias=True))
    def forward(self, x):
        out = F.softmax(self.classification(x), dim=1)
        return out, torch.argmax(out, dim=1)
class MakeDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}
def model(parameters_dict):
    # Loading parameters
    positive_classes = parameters_dict['positive_classes']
    positive_num = parameters_dict['positive_num']
    negtive_dataset = parameters_dict['negtive_dataset']
    binary_model_name = parameters_dict['binary_model_name']
    train_test_ratio = parameters_dict['train_test_ratio']
    lr = parameters_dict['learning_rate']
    epochs = parameters_dict['epochs']
    batch_size = parameters_dict['batch_size']
    # Loading binary model outputs
    binary_out_human = torch.load(rf'/home/xuyi/rotation1/final_ablation/wo_conv_attn/x_{binary_model_name}_0109_{str(positive_classes)}*{str(positive_num)}_{negtive_dataset}.pt').to(device)
    NP_human_family_index = torch.load(rf'/home/xuyi/rotation1/final_ablation/wo_conv_attn/y_{binary_model_name}_0109_{str(positive_classes)}*{str(positive_num)}_{negtive_dataset}.pt').to(device)
    # Train & test splitting in every class
    reps_train, reps_test = torch.zeros(1, 512).to(device), torch.zeros(1, 512).to(device)
    target_train, target_test = torch.zeros(1, positive_classes).to(device), torch.zeros(1, positive_classes).to(device)
    for class_id in range(positive_classes):
        train_set_index = torch.tensor(random.sample(range(0, positive_num), int(positive_num * train_test_ratio))).to(device)
        test_set_index = torch.tensor(list(set(list(range(0, positive_num))) - set(train_set_index.cpu().numpy().tolist()))).to(device)
        reps_train = torch.cat((reps_train, binary_out_human[class_id * positive_num : (class_id + 1) * positive_num, :].index_select(0, train_set_index)), dim=0)
        target_train = torch.cat((target_train, NP_human_family_index[class_id * positive_num : (class_id + 1) * positive_num, :].index_select(0, train_set_index)), dim=0)
        reps_test = torch.cat((reps_test, binary_out_human[class_id * positive_num : (class_id + 1) * positive_num, :].index_select(0, test_set_index)), dim=0)
        target_test = torch.cat((target_test, NP_human_family_index[class_id * positive_num : (class_id + 1) * positive_num, :].index_select(0, test_set_index)), dim=0)
    reps_train, reps_test = reps_train[1:, :], reps_test[1:, :]
    target_train, target_test = target_train[1:, :], target_test[1:, :]
    # Z-score
    reps_train = (reps_train - reps_train.mean(dim=1, keepdim=True)) / reps_train.std(dim=1, keepdim=True)
    reps_test = (reps_test - reps_test.mean(dim=1, keepdim=True)) / reps_test.std(dim=1, keepdim=True)
    # Training
    training_set = MakeDataset(reps_train, target_train)
    # print(f'Now begins training...')
    multiple_model = multiple_classification_model(input_dim=binary_out_human.shape[1], num_classes=positive_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(multiple_model.parameters(), lr=lr, momentum=0, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=False)
    multiple_model.train()
    for epoch in range(epochs):
        # print(f'epoch {epoch + 1}...', end='')
        start_time_epoch = time.time()
        dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
        total_loss = 0
        for batch in dataloader:
            x_train, y_train = batch['x'], batch['y']
            y_prob, _ = multiple_model(x_train)
            loss = criterion(y_prob, y_train)
            total_loss = total_loss + loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step(total_loss)
        avg_loss = round(total_loss.item() / reps_train.shape[0], 4)
        time_elapsed_epoch = round((time.time() - start_time_epoch) / 60, 2)
        # print(f'average loss is {Fore.RED}{avg_loss}{Fore.RESET}, time elapsed: {time_elapsed_epoch} min...')
    # print('Training ends, and now begins testing...')
    multiple_model.eval()
    _, y_pre = multiple_model(reps_test)
    y_true = torch.argmax(target_test, dim=1)
    micro_f1 = round(f1_score(y_true.cpu(), y_pre.cpu(), average='micro', zero_division=0.0), 4)
    macro_f1 = round(f1_score(y_true.cpu(), y_pre.cpu(), average='macro', zero_division=0.0), 4)
    acc = round(accuracy_score(y_true.cpu(), y_pre.cpu()), 4)
    _, _, f1, _ = precision_recall_fscore_support(y_true.cpu(), y_pre.cpu(), average=None, zero_division=0.0)
    recall = recall_score(y_true.cpu(), y_pre.cpu(), average=None, zero_division=0.0)
    precision = precision_score(y_true.cpu(), y_pre.cpu(), average=None, zero_division=0.0)
    confusions = confusion_matrix(y_true.cpu(), y_pre.cpu(), normalize='true')
    # np.save(rf'/home/xuyi/rotation1/multiple_model_results/confusions.npy', confusions)
    non_zero_f1 = np.round(np.mean(np.array(f1)[np.array(f1) != 0]), 4)
    print(f'Micro F1 score: {Fore.RED}{micro_f1}{Fore.RESET}')
    print(f'Macro F1 score: {Fore.RED}{macro_f1}{Fore.RESET}')
    print(f'Non-zero F1 score: {Fore.RED}{non_zero_f1}{Fore.RESET}')
    # print(f'Accuracy: {Fore.RED}{acc}{Fore.RESET}')
    # print(f1)
    # print(recall)
    # print(precision)
    # print(confusions)
    # torch.save(multiple_model.state_dict(), rf'/home/xuyi/rotation1/multiple_model_results/model.pth')
    return micro_f1, macro_f1, non_zero_f1
if __name__ == '__main__':
    # mF1, MF1, non = [], [], []
    # for b in range(16, 100):
    #     set_seed(seed=42)
    #     print(b)
    #     device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')
    #     params = {
	# 		"positive_classes": 20,
	# 		"positive_num": 90,
	# 		"negtive_dataset": 'allneg',
    #         "binary_model_name": 'e100b16_p512k3n1h4_BCESGD',
    #         "train_test_ratio": 0.8,
    #         "learning_rate": 1.5,
    #         "epochs": 80,
    #         "batch_size": b
    #     }
    #     micro_f1, macro_f1, non_zero_f1 = model(params)
    #     mF1.append(micro_f1)
    #     MF1.append(macro_f1)
    #     non.append(non_zero_f1)
    # print(np.max(np.array(mF1)), np.argmax(np.array(mF1)))
    # print(np.max(np.array(MF1)), np.argmax(np.array(MF1)))
    # print(np.max(np.array(non)), np.argmax(np.array(non)))
    set_seed(seed=2872) # 2872
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    params = {
			"positive_classes": 20,
			"positive_num": 90,
			"negtive_dataset": 'allneg',
            "binary_model_name": 'e100b16_p512k3n1h4_BCESGD',
            "train_test_ratio": 0.8,
            "learning_rate": 1.5,
            "epochs": 80,
            "batch_size": 69 # 69
    }
    micro_f1, macro_f1, non_zero_f1 = model(params)