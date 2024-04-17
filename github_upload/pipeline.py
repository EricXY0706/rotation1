import torch
import pandas as pd
from tqdm import *
from utils import set_seed, ESM_embedding
from model import binary_classification_model, multiple_classification_model
import argparse

def pipeline(input_file, binary_output_file, multiple_output_file, gpu_id):
	set_seed(seed=42)
	device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
	# Binary classification
	seq_list, token_representations, sequence_length = ESM_embedding(input_file, device)
	token_representations, sequence_length = token_representations.to(device), sequence_length.to(device)
	total_size, max_seq_length, emb_length = token_representations.shape[0], token_representations.shape[1], token_representations.shape[2]
	binary_model = binary_classification_model(max_seq_length=max_seq_length, emb_length=emb_length, proj_length=512, kernel_size=3, kernels=2, heads=4).to(device)
	binary_model.load_state_dict(torch.load(r'binary_model_parameters.pth'))
	binary_model.eval()
	print('Now begins binary classification...')
	cfs_index = {0: 'human', 1: 'non-human'}
	results = []
	binary_out, human_seqs = torch.zeros((1, 512)).to(device), []
	for seq_id in tqdm(range(total_size-1)):
		seq = seq_list[seq_id]
		x, l = token_representations[seq_id, :, :], sequence_length[seq_id]
		y_prob, y_pre, out = binary_model(x, l, device)
		cf = cfs_index[y_pre.item()]
		if cf == 'human':
			human_seqs.append(seq)
			binary_out = torch.cat((binary_out, out), dim=0)
		results.append([seq, cf, round(y_prob.item(), 4)])
	binary_out = binary_out[1:, :]
	# Save records to binary_output_file
	df = pd.DataFrame(results)
	df.columns = ['Sequence', 'Prediction', 'Probability']
	if binary_output_file.split('.')[-1] == 'csv':
		df.to_csv(rf'{str(binary_output_file)}', index=False)
	elif binary_output_file.split('.')[-1] == 'txt':
		df.to_csv(rf'{str(binary_output_file)}', sep='\t', index=False)
	print(f'Binary classification finished! See results at {str(binary_output_file)}.')
	# Multiple classification
	user_input = 'y'
	if multiple_output_file == None:
		user_input = input('Continue to multiple classification? [y/n]:').lower()
	if user_input == 'y' or user_input == 'yes':
		print('Now begins multiple classification...')
		multiple_model = multiple_classification_model(input_dim=binary_out.shape[1], num_classes=20).to(device)
		multiple_model.load_state_dict(torch.load(r'multiple_model_parameters.pth'))
		multiple_model.eval()
		cfs_index = ['Bombesin/neuromedin-B/ranatensin', 'Calcitonin', 'Chromogranin/secretogranin', 'FMRFamide related peptide', 'Galanin', 'Gastrin/cholecystokinin', 'Glucagon', 'GnRH', 'Insulin', 'Others', 'NPY', 'Natriuretic peptide', 'Opioid', 'POMC', 'RFamide neuropeptide', 'Sauvagine/corticotropin-releasing factor/urotensin I', 'Serpin', 'Somatostatin', 'Tachykinin', 'Vasopressin/oxytocin']
		y_prob, y_pre = multiple_model(binary_out)
		results = []
		for i, seq in enumerate(human_seqs):
			results.append([seq, cfs_index[y_pre[i].item()], round(torch.max(y_prob[i]).item(), 4)])
		# Save records to multiple_output_file
		df = pd.DataFrame(results)
		df.columns = ['Sequence', 'Prediction', 'Probability']
		output_route = multiple_output_file if multiple_output_file != None else './multiple_prediction_results.csv'
		if output_route.split('.')[-1] == 'csv':
			df.to_csv(rf'{str(output_route)}', index=False)
		elif output_route.split('.')[-1] == 'txt':
			df.to_csv(rf'{str(output_route)}', sep='\t', index=False)
		print(f'Multiple classification finished! See results at {str(output_route)}.')
	elif user_input == 'n' or user_input == 'no':
		pass
	print('Pipeline finished.')

if __name__ == '__main__':
    # Args config
    parser = argparse.ArgumentParser(prog='NPred\n', usage='python pipeline.py [-h] -i InputFile -bo BinaryOutputFile [-mo MultipleOutputFile] -g GPUID', description='A silly pipeline for Neuropeptide prediction and functional annotation :)')
    parser.add_argument('-i', '--input', help='Input FASTA file', required=True)
    parser.add_argument('-bo', '--binary_output', help='Output file of binary classification in csv or txt format', required=True)
    parser.add_argument('-mo', '--multiple_output', help='Output file of multiple classification in csv or txt format', required=False)
    parser.add_argument('-g', '--gpu', help='GPU id', required=True)
    args = parser.parse_args()
    InputFile = args.input
    BinaryOutputFile = args.binary_output
    MultipleOutputFile = args.multiple_output
    GPU_ID = args.gpu
    # Classification
    pipeline(InputFile, BinaryOutputFile, MultipleOutputFile, GPU_ID)