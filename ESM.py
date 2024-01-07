import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
from einops import rearrange
import esm
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import *
import random
from sklearn import manifold
from colorama import Fore
import os
import wandb
import time

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def data_loader():
    NP_table_human = pd.read_csv(r'/tmp/pycharm_project_996/datasets/NeuroPep2.0_7-100_human_family_60.csv')
    NP_table_others = pd.read_csv(r'/tmp/pycharm_project_996/datasets/NeuroPep2.0_7-100_others_family.csv')
    NP_human_list = NP_table_human['Sequence'].tolist()
    NP_others_list = NP_table_others['Sequence'].tolist()
    NP_tag = ['human'] * len(NP_human_list)
    NP_tag.extend(['others'] * len(NP_others_list))
    return NP_human_list, NP_others_list, NP_tag
def ESM_embedding(ESM_model_name):
    ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    ESM_model.eval()
    NP_human_list, NP_others_list, NP_tag = data_loader()
    NP_list = NP_human_list
    NP_list.extend(NP_others_list)
    data = []
    for i in range(len(NP_list)):
        data.append(("protein"+str(i+1), str(NP_list[i])))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_size, ESM_layers = 32, 33
    print('ESM model embedding...')
    with torch.no_grad():
        first_sample_tokens = batch_tokens[0:2]
        first_sample_result = ESM_model(first_sample_tokens, repr_layers=[ESM_layers], return_contacts=True)
    token_representations = first_sample_result["representations"][ESM_layers]
    with torch.no_grad():
        for i in tqdm(range(2, len(batch_tokens), batch_size)):
            tiny_batch_tokens = batch_tokens[i:i+batch_size]
            tiny_batch_result = ESM_model(tiny_batch_tokens, repr_layers=[ESM_layers], return_contacts=True)
            tiny_batch_representations = tiny_batch_result["representations"][ESM_layers]
            token_representations = torch.cat((token_representations, tiny_batch_representations), dim=0)
    sequence_representations, sequence_length = [], []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1:tokens_len-1].mean(0))
        sequence_length.append(tokens_len.item()-2)
    # for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
    #     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
    #     plt.title(seq)
    #     plt.show()
    print('ESM model embedding finished!')
    torch.save(token_representations, rf'/tmp/pycharm_project_996/ESM_embeddings/tokens/tokens_{ESM_model_name}.pt')
    torch.save(torch.tensor(sequence_length), rf'/tmp/pycharm_project_996/ESM_embeddings/seq_length/seq_length_{ESM_model_name}.pt')
    return sequence_representations, NP_tag, torch.tensor(sequence_length)
def tsne_embedding(ESM_model_name):
    sequence_representations, NP_tag, sequence_length = ESM_embedding(ESM_model_name)
    print('t-SNE algorithm embedding...')
    seq_reps = sequence_representations[0].numpy()
    seq_reps = np.expand_dims(seq_reps, axis=0)
    for i in range(1, len(sequence_representations)):
        seq_rep = sequence_representations[i].numpy()
        seq_rep = np.expand_dims(seq_rep, axis=0)
        seq_reps = np.concatenate((seq_reps, seq_rep), axis=0)
    x_max, x_min = np.max(seq_reps, 1), np.min(seq_reps, 1)
    for i in range((seq_reps.shape[1])):
        seq_reps[:, i] = (seq_reps[:, i] - x_min) / (x_max - x_min)
    TSNE_transfer = manifold.TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, init='pca', random_state=42)
    seq_reps_tsne_emb = TSNE_transfer.fit_transform(seq_reps)
    plt.figure()
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
     'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
     'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta',
     'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
     'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick',
     'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
     'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush',
     'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen',
     'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue',
     'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
     'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
     'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
     'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
     'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
     'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow',
     'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
     'whitesmoke', 'yellow', 'yellowgreen']
    seq_reps_tsne_emb_human_x, seq_reps_tsne_emb_human_y = [], []
    seq_reps_tsne_emb_others_x, seq_reps_tsne_emb_others_y = [], []
    for seq_rep_id in range(seq_reps_tsne_emb.shape[0]):
        seq_rep_tsne_emb = seq_reps_tsne_emb[seq_rep_id]
        seq_family = NP_tag[seq_rep_id]
        if seq_family == 'human':
            seq_reps_tsne_emb_human_x.append(seq_rep_tsne_emb[0])
            seq_reps_tsne_emb_human_y.append(seq_rep_tsne_emb[1])
        else:
            seq_reps_tsne_emb_others_x.append(seq_rep_tsne_emb[0])
            seq_reps_tsne_emb_others_y.append(seq_rep_tsne_emb[1])
    plt.scatter(seq_reps_tsne_emb_human_x, seq_reps_tsne_emb_human_y, s=10, color='rosybrown', alpha=0.7, label='Human')
    plt.scatter(seq_reps_tsne_emb_others_x, seq_reps_tsne_emb_others_y, s=10, color='royalblue', alpha=0.7, label='Others')
    plt.title('t-SNE embedding of peptides in human or others after augmentation')
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.legend()
    plt.savefig(rf'/tmp/pycharm_project_996/ESM-tSNE/t-SNE_aug_human_vs_others_{ESM_model_name}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    print('t-SNE algorithm embedding finished!')
class binary_classification_model(nn.Module):
    def __init__(self, max_seq_length, emb_length, proj_length, kernel_size, kernels, heads):
        super(binary_classification_model, self).__init__()
        self.max_seq_length = max_seq_length
        self.proj_length = proj_length
        self.kernels = kernels
        self.heads = heads
        self.projection = nn.Linear(emb_length, proj_length, bias=True)
        self.conv = nn.Conv1d(in_channels=proj_length, out_channels=proj_length, kernel_size=kernel_size, padding='valid')
        self.attn = nn.Linear(proj_length, heads * max_seq_length, bias=True)
        self.classification = nn.Sequential(nn.Linear(proj_length, int(proj_length / 2), bias=True),
                                            nn.LeakyReLU(),
                                            nn.Linear(int(proj_length / 2), 2, bias=True))
    def forward(self, x, seq_length):
        # x: [batch_size, max_seq_length, emb_length] -> [b, s, e]
        x = rearrange(x, 'b s e -> (b s) e')
        representations_proj = self.projection(x) # [(b * s), e] -> [(b * s), p]
        representations_proj = rearrange(representations_proj, '(b s) p -> b p s', s=self.max_seq_length)
        representations_conv = torch.zeros(1, self.proj_length, self.max_seq_length)
        for i, l in enumerate(seq_length):
            l = l.item()
            representation_proj = representations_proj[i, :, 1:l+1]
            # representations_proj = F.relu(representations_proj)
            representation_proj = rearrange(representation_proj, 'p l -> () p l')
            out = F.relu(self.conv(representation_proj), inplace=False)
            for k in range(self.kernels - 1):
                # TODO
                # nn.init.kaiming_normal_(self.conv.weight)
                out = out + F.relu(self.conv(representation_proj), inplace=False)
            out /= self.kernels # [1, p, l-2]
            out = rearrange(out, '1 p m -> p m') # [p, l-2]
            out = torch.cat((torch.zeros(out.shape[0], 1), out), dim=1)
            out = torch.cat((out, torch.zeros(out.shape[0], (self.max_seq_length - l + 1))), dim=1) # [p, s]
            out = rearrange(out, 'p s -> () p s')
            representations_conv = torch.cat((representations_conv, out), dim=0)
        representations_conv = rearrange(representations_conv[1:, :, :], 'b p s -> (b s) p')
        attn = F.softmax(self.attn(representations_conv), dim=1) # [(b * s), (h * s)]
        attn = rearrange(attn, 't (h s) -> h t s', h=self.heads) # [h, (b * s), s]
        representations_attn = torch.zeros(1, self.proj_length)
        for i in range(seq_length.shape[0]):
            attn_score = attn[:, i * self.max_seq_length:(i+1) * self.max_seq_length, :] # [h, s, s]
            attn_score = rearrange(attn_score, 'h p q -> (h p) q') # [(h * s), s]
            representation_attn = representations_conv[i * self.max_seq_length:(i+1) * self.max_seq_length, :] # [s, p]
            representation_attn = torch.matmul(attn_score, representation_attn) # [(h s), p]
            representation_attn = rearrange(representation_attn, '(h s) p -> h s p', h=self.heads)
            representation_attn = torch.mean(representation_attn, dim=0) # [s, p]
            representation_attn = torch.sum(representation_attn, dim=0) # [p]
            representation_attn = rearrange(representation_attn, 'p -> () p')
            representations_attn = torch.cat((representations_attn, representation_attn), dim=0)
        representations_attn = representations_attn[1:, :] # [b, p]
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
    training_test_ratio = parameters_dict['training_ratio']
    epochs = parameters_dict['epochs']
    kfolds = parameters_dict['kfolds']
    batch_size = parameters_dict['batch_size']
    proj_length = parameters_dict['projection_length']
    kernel_size = parameters_dict['kernel_size']
    n_kernels = parameters_dict['n_kernels']
    n_heads = parameters_dict['n_heads']
    lr = parameters_dict['learning_rate']
    token_representations = torch.load(rf'/tmp/pycharm_project_996/ESM_embeddings/tokens/tokens_{ESM_model_name}.pt')
    sequence_length = torch.load(rf'/tmp/pycharm_project_996/ESM_embeddings/seq_length/seq_length_{ESM_model_name}.pt')
    _, _, NP_tag = data_loader()
    total_size, max_seq_length, emb_length = token_representations.shape[0], token_representations.shape[1], token_representations.shape[2]
    target_representations = torch.zeros(total_size, 2)
    for i, tag in enumerate(NP_tag):
        if tag == 'human':
            target_representations[i, 0] = 1
        else:
            target_representations[i, 1] = 1
    training_index = torch.tensor(random.sample(range(0, total_size), int(total_size * training_test_ratio)))
    testing_index = torch.tensor(list(set(list(range(0, total_size))) - set(training_index)))
    reps_train, reps_test = token_representations.index_select(0, training_index), token_representations.index_select(0, testing_index)
    target_train, target_test = target_representations.index_select(0, training_index), target_representations.index_select(0, testing_index)
    length_train, length_test = sequence_length.index_select(0, training_index), sequence_length.index_select(0, testing_index)
    dataset = MakeDataset(reps_train, target_train, length_train)
    colors = ['#457479', '#432B3E', '#D64B3F', '#E5855F', '#EED5B9', '#853F29']
    results, AUCs = [], []
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    print('Now begins training...')
    plt.figure()
    if not os.path.exists(rf'/tmp/pycharm_project_996/binary_model_results/CV/model/{result_name}'):
        os.mkdir(rf'/tmp/pycharm_project_996/binary_model_results/CV/model/{result_name}')
    for k, (train_index, val_index) in enumerate(kf.split(dataset)):
        print(f'\nNow training No.{Fore.RED}{k + 1}{Fore.RESET} fold...')
        binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length,
                                                   proj_length=proj_length, kernel_size=kernel_size, kernels=n_kernels, heads=n_heads)
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
        confusions = confusion_matrix(torch.max(val_y, dim=1).indices.numpy(), y_pre.numpy())
        TP, FP, TN, FN = confusions[0][0], confusions[0][1], confusions[1][0], confusions[1][1]
        ACC = round((TP + TN) / (TP + FN + TN + FP), 4)
        Precision = round(TP / (TP + FP), 4)
        Recall = round(TP / (TP + FN), 4)
        Sensitivity = round(TP / (TP + FN), 4)
        Specificity = round(TN / (FP + TN), 4)
        MCC = round((TP * TN - FP * FN) / pow((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 0.5), 4)
        F1_score = round(2 * Precision * Recall / (Precision + Recall), 4)
        time_elapsed_kfold = round((time.time() - start_time_kfold) / 60, 2)
        print(f'\nFold {k + 1}, training results on validation-set {k + 1}:')
        print(f'AUC: {Fore.RED}{AUC}{Fore.RESET}')
        print(f'Accuracy: {Fore.RED}{ACC}{Fore.RESET}')
        print(f'Precision: {Fore.RED}{Precision}{Fore.RESET}')
        print(f'Recall: {Fore.RED}{Recall}{Fore.RESET}')
        print(f'Sensitivity: {Fore.RED}{Sensitivity}{Fore.RESET}')
        print(f'Specificity: {Fore.RED}{Specificity}{Fore.RESET}')
        print(f'Matthews coefficient: {Fore.RED}{MCC}{Fore.RESET}')
        print(f'F1-score: {Fore.RED}{F1_score}{Fore.RESET}')
        print(f'Time elapsed: {Fore.RED}{time_elapsed_kfold}{Fore.RESET} min')
        # Record
        AUCs.append(AUC)
        plt.plot(fpr, tpr, color=colors[k], lw=2, label=f'fold{k}')
        results.append([k, AUC, ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score])
        torch.save(binary_model.state_dict(), rf'/tmp/pycharm_project_996/binary_model_results/CV/model/{result_name}/k={k+1}.pth')
    print('Training ends, and now begins testing...')
    binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length,
                                               proj_length=proj_length, kernel_size=kernel_size, kernels=n_kernels, heads=n_heads)
    best_model_index = np.argmax(np.array(AUCs)) + 1
    best_model_path = rf'/tmp/pycharm_project_996/binary_model_results/CV/model/{result_name}/k={best_model_index}.pth'
    binary_model.load_state_dict(torch.load(best_model_path))
    binary_model.eval()
    y_prob, y_pre = binary_model(reps_test, length_test)
    fpr, tpr, thresholds = roc_curve(target_test.numpy()[:, 0], y_prob.detach().numpy()[:, 0], pos_label=1)
    AUC = round(auc(fpr, tpr), 4)
    confusions = confusion_matrix(torch.max(target_test, dim=1).indices.numpy(), y_pre.numpy())
    TP, FP, TN, FN = confusions[0][0], confusions[0][1], confusions[1][0], confusions[1][1]
    ACC = round((TP + TN) / (TP + FN + TN + FP), 4)
    Precision = round(TP / (TP + FP), 4)
    Recall = round(TP / (TP + FN), 4)
    Sensitivity = round(TP / (TP + FN), 4)
    Specificity = round(TN / (FP + TN), 4)
    MCC = round((TP * TN - FP * FN) / pow((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN), 0.5), 4)
    F1_score = round(2 * Precision * Recall / (Precision + Recall), 4)
    print(f'AUC: {Fore.RED}{AUC}{Fore.RESET}')
    print(f'Accuracy: {Fore.RED}{ACC}{Fore.RESET}')
    print(f'Precision: {Fore.RED}{Precision}{Fore.RESET}')
    print(f'Recall: {Fore.RED}{Recall}{Fore.RESET}')
    print(f'Sensitivity: {Fore.RED}{Sensitivity}{Fore.RESET}')
    print(f'Specificity: {Fore.RED}{Specificity}{Fore.RESET}')
    print(f'Matthews coefficient: {Fore.RED}{MCC}{Fore.RESET}')
    print(f'F1-score: {Fore.RED}{F1_score}{Fore.RESET}')
    # Record
    plt.plot(fpr, tpr, color=colors[-1], lw=2, label=f'testing set')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('FPR', fontsize=20)
    plt.ylabel('TPR', fontsize=20)
    plt.legend()
    plt.savefig(rf'/tmp/pycharm_project_996/binary_model_results/ROC/{result_name}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    results.append(['testing set', AUC, ACC, Precision, Recall, Sensitivity, Specificity, MCC, F1_score])
    result_df = pd.DataFrame(results)
    result_df.columns = ['Dataset', 'AUC', 'Accuracy', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'MCC', 'F1']
    result_df.to_csv(rf'/tmp/pycharm_project_996/binary_model_results/result/{result_name}.csv', index=False)
if __name__ == '__main__':
    # ESM_embedding(ESM_model_name='esm2_t33_650M_UR50D')
    # tsne_embedding(ESM_model_name='esm2_t33_650M_UR50D')
    wandb.init(project='binary_classification', entity='team-xy')
    params = {"training_ratio": 0.8,
              "epochs": 15,
              "kfolds": 5,
              "batch_size": 8,
              "projection_length": 512,
              "kernel_size": 3,
              "n_kernels": 2,
              "n_heads": 2,
              "learning_rate": 1e-2
              }
    project_name = 'e' + str(params['epochs']) + 'k' + str(params['kfolds']) + 'b' + str(params['batch_size']) + '_p' + \
                  str(params['projection_length']) + 'k' + str(params['kernel_size']) + 'n' + str(params['n_kernels']) + \
                  'h' + str(params['n_heads']) + '_BCEWithLogitsSGD'
    print(f'project name: {project_name}')
    model(ESM_model_name='esm2_t33_650M_UR50D', result_name=project_name, parameters_dict=params)
    pass