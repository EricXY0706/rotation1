import torch
from utils import set_seed, data_loader
from ESM_GPU import binary_classification_model
def load_human_reps(model_name, pos_name, neg_name):
    token_representations = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/tokens/tokens_esm2_t33_650M_UR50D_{pos_name}_{neg_name}.pt').to(device)
    sequence_length = torch.load(rf'/home/xuyi/rotation1/ESM_embeddings/seq_length/seq_length_esm2_t33_650M_UR50D_{pos_name}_{neg_name}.pt').to(device)
    max_seq_length, emb_length = token_representations.shape[1], token_representations.shape[2]
    model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length, proj_length=512, kernel_size=3, kernels=2, heads=4).to(device)
    model.load_state_dict(torch.load(rf'/home/xuyi/rotation1/binary_model_results/model/b_{model_name}_{pos_name}_{neg_name}.pth'))
    NP_human_list, _, _, NP_human_family_list = data_loader()
    _, _, binary_out = model(token_representations, sequence_length, device)
    binary_out_human = binary_out[:len(NP_human_list), :]
    NP_human_family = list(set(NP_human_family_list))
    NP_human_family.sort()
    NP_human_family_index = torch.zeros(len(NP_human_family_list), len(NP_human_family))
    for i, NP_family in enumerate(NP_human_family_list):
        NP_human_family_index[i, NP_human_family.index(NP_family)] = 1
    torch.save(binary_out_human, rf'/home/xuyi/rotation1/binary_embeddings/x_{model_name}_{pos_name}_{neg_name}.pt')
    torch.save(NP_human_family_index, rf'/home/xuyi/rotation1/binary_embeddings/y_{model_name}_{pos_name}_{neg_name}.pt')
    print(binary_out_human.shape)
    print(NP_human_family_index.shape)
    return binary_out_human, NP_human_family_index, NP_human_family
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_name = 'e100b16_p512k3n2h4_BCESGD'
pos_name, neg_name = '20*85+merge', 'allneg'
load_human_reps(model_name=model_name, pos_name=pos_name, neg_name=neg_name)