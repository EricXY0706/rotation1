import pandas as pd
import random
import os
from collections import Counter
from colorama import Fore
from augmentation_utils import combined_aug
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
def load_data():
    NPs_human = pd.read_csv(rf'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_0313.csv')
    # NPs_human = NPs[NPs['human'] == 'human_fam']
    NPs_human_families = list(set(NPs_human['Family'].tolist()))
    NPs_human_families.sort()
    return NPs_human, NPs_human_families
def aug_or_sub(uniform_num):
	NPs_human, NPs_human_families = load_data()
	result_df = pd.DataFrame()
	result_seqs, result_family, result_length = [], [], []
	for family in NPs_human_families:
		NPs_human_family = NPs_human[NPs_human['Family'] == family]
		NPs_human_family_seqs = NPs_human_family['Sequence'].tolist()
		if len(NPs_human_family_seqs) < uniform_num:
			print(f'For family {Fore.RED}{family}{Fore.RESET}, '
				f'numbers: {Fore.RED}{len(NPs_human_family_seqs)}{Fore.RESET} -> ', end='')
			NPs_human_family_seqs_amplified, count = combined_aug(family, NPs_human_family_seqs, p=0.5, num_amplification=uniform_num)
			print(f'{Fore.RED}{len(NPs_human_family_seqs_amplified)}{Fore.RESET}, '
				f'total amplification rounds: {Fore.RED}{count}{Fore.RESET}')
			result_seqs.extend(NPs_human_family_seqs_amplified)
			result_family.extend([family] * uniform_num)
		else:
			print(f'For family {Fore.RED}{family}{Fore.RESET}, '
				f'numbers: {Fore.RED}{len(NPs_human_family_seqs)}{Fore.RESET}, '
				f'no need to do the augmentation')
			NPs_human_family_seqs_undersampled = []
			undersample_index = random.sample(range(0, len(NPs_human_family_seqs)), uniform_num)
			for index in undersample_index:
				NPs_human_family_seqs_undersampled.append(NPs_human_family_seqs[index])
			result_seqs.extend(NPs_human_family_seqs_undersampled)
			result_family.extend([family] * uniform_num)
	for seq in result_seqs:
		result_length.append(len(seq))
	result_df['Sequence'] = pd.Series(result_seqs)
	result_df['Family'] = pd.Series(result_family)
	result_df['Length'] = pd.Series(result_length)
	result_df.to_csv(rf'/home/xuyi/rotation1/datasets/NeuroPep2.0_7-100_human_family_0313_25*{uniform_num}.csv', index=False)
	print('Augmentation finished and the dataframe has beem saved!')
if __name__ == '__main__':
    aug_or_sub(70)