import torch
import random
import os
import numpy as np
import pandas as pd
from Bio import SeqIO
import esm
from tqdm import *

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def data_loader(input_file):
	input_file_route = rf'{str(input_file)}'
	seq_list = []
	if input_file.split('.')[-1] == 'fasta':
		for record in SeqIO.parse(input_file_route, 'fasta'):
			seq = str(record.seq).strip('\n').upper()
			seq_list.append(seq)
	elif input_file.split('.')[-1] == 'csv':
		input_file = pd.read_csv(input_file_route)
		for index, row in input_file.iterrows():
			seq = row.Sequence.strip('\n').upper()
			seq_list.append(seq)
	return seq_list
def ESM_embedding(input_file, device):
    print('ESM model embedding...')
    batch_size, ESM_layers = 32, 33
    ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    ESM_model.to(device)
    batch_converter = alphabet.get_batch_converter()
    ESM_model.eval()
    seq_list = data_loader(input_file)
    data = []
    for i in range(len(seq_list)):
        data.append(("protein"+str(i+1), str(seq_list[i])))
    data.append(("Control", 'C'*100))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
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
    print('ESM model embedding finished!')
    return seq_list, token_representations, torch.tensor(sequence_length)