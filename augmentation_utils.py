import random

def replacement_dictionary(NP_seqs, p):
    rep_dict = [['A', 'V'], ['S', 'T'], ['F', 'Y'], ['K', 'R'], ['C', 'M'], ['D', 'E'], ['N', 'Q'], ['V', 'I']]
    probability0 = random.random()
    NP_seqs_plus = []
    if probability0 > p:
        NP_index = random.sample(range(0, len(NP_seqs)), round(len(NP_seqs) * probability0))
        for NP_id in NP_index:
            for rep in rep_dict:
                probability1 = random.random()
                if probability1 > p:
                    new_seq = NP_seqs[NP_id].replace(rep[0], rep[1])
                    NP_seqs_plus.append(new_seq)
                else:
                    pass
    else:
        pass
    return set(NP_seqs_plus)
def replacement_alanine(NP_seqs, p):
    probability0 = random.random()
    NP_seqs_plus = []
    if probability0 > p:
        NP_index = random.sample(range(0, len(NP_seqs)), round(len(NP_seqs) * probability0))
        for NP_id in NP_index:
            amino_index = random.sample(range(0, len(NP_seqs[NP_id])), round(len(NP_seqs[NP_id]) * (1 - probability0)))
            new_seq = list(NP_seqs[NP_id])
            for amino_id in amino_index:
                new_seq[amino_id] = 'A'
            new_seq = ''.join(new_seq)
            NP_seqs_plus.append(new_seq)
    else:
        pass
    return set(NP_seqs_plus)
def global_random_shuffling(NP_seqs, p):
    probability0 = random.random()
    NP_seqs_plus = []
    if probability0 > p:
        NP_index = random.sample(range(0, len(NP_seqs)), round(len(NP_seqs) * probability0))
        for NP_id in NP_index:
            str_list = list(NP_seqs[NP_id])
            random.shuffle(str_list)
            NP_seqs_plus.append(''.join(str_list))
    else:
        pass
    return set(NP_seqs_plus)
def local_seq_shuffling(NP_seqs, p):
    probability0 = random.random()
    NP_seqs_plus = []
    if probability0 > p:
        NP_index = random.sample(range(0, len(NP_seqs)), round(len(NP_seqs) * probability0))
        for NP_id in NP_index:
            seq1 = NP_seqs[NP_id][:round((probability0 - p) * len(NP_seqs[NP_id]))]
            seq2 = NP_seqs[NP_id][round((probability0 - p) * len(NP_seqs[NP_id])):-round((probability0 - p) * len(NP_seqs[NP_id]))]
            seq3 = NP_seqs[NP_id][-round((probability0 - p) * len(NP_seqs[NP_id])):]
            seq2_str_list = list(seq2)
            random.shuffle(seq2_str_list)
            new_seq = seq1 + ''.join(seq2_str_list) + seq3
            NP_seqs_plus.append(new_seq)
    else:
        pass
    return set(NP_seqs_plus)
def sequence_reversion(NP_seqs, p):
    probability0 = random.random()
    NP_seqs_plus = []
    if probability0 > p:
        NP_index = random.sample(range(0, len(NP_seqs)), round(len(NP_seqs) * probability0))
        for NP_id in NP_index:
            new_seq = NP_seqs[NP_id][::-1]
            NP_seqs_plus.append(new_seq)
    else:
        pass
    return set(NP_seqs_plus)
def subsampling(NP_seqs, p):
    probability0 = random.random()
    NP_seqs_plus = []
    if probability0 > p:
        NP_index = random.sample(range(0, len(NP_seqs)), round(len(NP_seqs) * probability0))
        for NP_id in NP_index:
            new_seq = NP_seqs[NP_id][round(len(NP_seqs[NP_id]) * (probability0 - p)):-round(len(NP_seqs[NP_id]) * (probability0 - p))]
            NP_seqs_plus.append(new_seq)
    else:
        pass
    return set(NP_seqs_plus)
def check_length(NP_seqs):
    result = []
    for seq in NP_seqs:
        if 7 <= len(seq) <= 100:
            result.append(seq)
    return result
def combined_aug(NP_family, NP_seqs, p=0.5, num_amplification=50):
    amplified_NP_seqs = NP_seqs
    augmented_NP_seqs = NP_seqs
    count = 0
    while len(amplified_NP_seqs) < num_amplification:
        globals()['NP_seqs_plus_1'] = replacement_dictionary(amplified_NP_seqs, p)
        globals()['NP_seqs_plus_2'] = replacement_alanine(amplified_NP_seqs, p)
        globals()['NP_seqs_plus_3'] = global_random_shuffling(amplified_NP_seqs, p)
        globals()['NP_seqs_plus_4'] = local_seq_shuffling(amplified_NP_seqs, p)
        globals()['NP_seqs_plus_5'] = sequence_reversion(amplified_NP_seqs, p)
        globals()['NP_seqs_plus_6'] = subsampling(amplified_NP_seqs, p)
        for i in range(1, 7):
            amplified_NP_seqs = set(amplified_NP_seqs) | globals()['NP_seqs_plus_'+str(i)]
        amplified_NP_seqs = list(amplified_NP_seqs)
        amplified_NP_seqs = check_length(amplified_NP_seqs)
        count += 1
    amplification_set = list(set(amplified_NP_seqs) - set(NP_seqs))
    amplification_index = random.sample(range(0, len(amplification_set)), (num_amplification-len(NP_seqs)))
    for index in amplification_index:
        augmented_NP_seqs.append(amplification_set[index])
    return augmented_NP_seqs, count