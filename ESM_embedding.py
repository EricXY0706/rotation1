import esm
import torch
from tqdm import *
from utils import data_loader

def ESM_embedding(ESM_model_name):
    ESM_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    ESM_model.eval()
    NP_human_list, NP_others_list, NP_tag, NP_family_list = data_loader()
    NP_list = NP_human_list
    NP_list.extend(NP_others_list)
    data = []
    for i in range(len(NP_human_list)):
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
    torch.save(token_representations, rf'/home/xuyi/rotation1/ESM_embeddings/tokens/tokens_{ESM_model_name}_0313_25*70_allneg.pt')
    torch.save(torch.tensor(sequence_length), rf'/home/xuyi/rotation1/ESM_embeddings/seq_length/seq_length_{ESM_model_name}_0313_25*70_allneg.pt')
    return sequence_representations, NP_tag, torch.tensor(sequence_length), NP_family_list
ESM_embedding('esm2_t33_650M_UR50D')