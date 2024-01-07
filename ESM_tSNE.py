from ESM_embedding import ESM_embedding
from sklearn import manifold
from matplotlib import pyplot as plt
import numpy as np

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
    plt.savefig(rf'/home/xuyi/rotation1/ESM-tSNE/t-SNE_aug_human_vs_others_{ESM_model_name}.tiff', dpi=300, bbox_inches='tight')
    plt.show()
    print('t-SNE algorithm embedding finished!')