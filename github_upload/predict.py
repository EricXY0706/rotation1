import torch
import pandas as pd
from tqdm import *
from utils import set_seed, ESM_embedding
from model import binary_classification_model
import argparse

def model(input_file, output_file, gpu_id):
	set_seed(seed=42)
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	seq_list, token_representations, sequence_length = ESM_embedding(input_file, device)
	token_representations, sequence_length = token_representations.to(device), sequence_length.to(device)
	total_size, max_seq_length, emb_length = token_representations.shape[0], token_representations.shape[1], token_representations.shape[2]
	binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length, proj_length=512, kernel_size=3, kernels=1, heads=4).to(device)
	binary_model.load_state_dict(torch.load(r'binary_model_parameters.pth'))
	binary_model.eval()
	print('Now begins prediction...')
	cfs_index = {0: 'human', 1: 'non-human'}
	results = []
	for seq_id in tqdm(range(total_size-1)):
		seq = seq_list[seq_id]
		x, l = token_representations[seq_id, :, :], sequence_length[seq_id]
		y_prob, y_pre = binary_model(x, l, device)
		cf = cfs_index[y_pre.item()]
		results.append([seq, cf, round(y_prob.item(), 4)])
	df = pd.DataFrame(results)
	df.columns = ['Sequence', 'Prediction', 'Probability']
	if output_file.split('.')[-1] == 'csv':
		df.to_csv(rf'{str(output_file)}', index=False)
	elif output_file.split('.')[-1] == 'txt':
		df.to_csv(rf'{str(output_file)}', sep='\t', index=False)
	print(f'Prediction finished! See results at {str(output_file)}.')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NPred')
    parser.add_argument('-i', '--input', help='Input FASTA file', required=True)
    parser.add_argument('-o', '--output', help='Output file in csv or txt format', required=True)
    parser.add_argument('-g', '--gpu', help='GPU id', required=True)
    args = parser.parse_args()
    InputFile = args.input
    OutputFile = args.output
    GPU_ID = args.gpu
    model(InputFile, OutputFile, GPU_ID)