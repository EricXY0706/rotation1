from ESM_embedding import ESM_embedding
from sklearn import manifold
from matplotlib import pyplot as plt
import numpy as np

def tsne_embedding(ESM_model_name):
    y =  np.load(r'/home/xuyi/rotation1/ESM-tSNE/binary_y.npy')
    for type in ['input', 'projection', 'convolution', 'attention']:
        reps = np.load(rf'/home/xuyi/rotation1/ESM-tSNE/binary_test_{type}.npy')
        x_mean, x_std = np.mean(reps, axis=0), np.std(reps, axis=0)
        reps = (reps - x_mean) / x_std
        tsne = manifold.TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, init='pca', random_state=42)
        reps_tsne = tsne.fit_transform(reps)
        reps_tsne_human_x, reps_tsne_human_y = [], []
        reps_tsne_others_x, reps_tsne_others_y = [], []
        for rep_id in range(reps_tsne.shape[0]):
            if y[rep_id] == 1:
                reps_tsne_human_x.append(reps_tsne[rep_id][0])
                reps_tsne_human_y.append(reps_tsne[rep_id][1])
            else:
                reps_tsne_others_x.append(reps_tsne[rep_id][0])
                reps_tsne_others_y.append(reps_tsne[rep_id][1])
        plt.figure()
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = 'Arial'
        plt.scatter(reps_tsne_human_x, reps_tsne_human_y, s=15, color='royalblue', alpha=0.7, label='Human')
        plt.scatter(reps_tsne_others_x, reps_tsne_others_y, s=15, color='rosybrown', alpha=0.7, label='Others')
        plt.xlabel('dim1')
        plt.ylabel('dim2')
        plt.legend(loc='upper right')
        plt.savefig(rf'/home/xuyi/rotation1/ESM-tSNE/t-SNE_binary_testset_after_{type}.pdf', dpi=300, bbox_inches='tight')
        print(f't-SNE algorithm embedding finished for {type}!')
    # sequence_representations, NP_tag, sequence_length, fam_list = ESM_embedding(ESM_model_name)
    # print('t-SNE algorithm embedding...')
    # seq_reps = sequence_representations[0].numpy()
    # seq_reps = np.expand_dims(seq_reps, axis=0)
    # for i in range(1, len(sequence_representations)):
    #     seq_rep = sequence_representations[i].numpy()
    #     seq_rep = np.expand_dims(seq_rep, axis=0)
    #     seq_reps = np.concatenate((seq_reps, seq_rep), axis=0)
    # x_mean, x_std = np.mean(seq_reps, axis=0), np.std(seq_reps, axis=0)
    # seq_reps = (seq_reps - x_mean) / x_std
    # TSNE_transfer = manifold.TSNE(n_components=2, perplexity=50, learning_rate=200, n_iter=1000, init='pca', random_state=42)
    # seq_reps_tsne_emb = TSNE_transfer.fit_transform(seq_reps)
    # np.save(r'/home/xuyi/rotation1/ESM-tSNE/fams.npy', seq_reps_tsne_emb)
    # plt.figure()
    # plt.rcParams['font.family'] = 'sans-serif'
    # plt.rcParams['font.sans-serif'] = 'Arial'
    # colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
    #  'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk',
    #  'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta',
    #  'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue',
    #  'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 'firebrick',
    #  'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 'green',
    #  'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 'khaki', 'lavender', 'lavenderblush',
    #  'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgreen',
    #  'lightgray', 'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightsteelblue',
    #  'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid',
    #  'mediumpurple', 'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred',
    #  'midnightblue', 'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab',
    #  'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip',
    #  'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'red', 'rosybrown', 'royalblue', 'saddlebrown',
    #  'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'snow',
    #  'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white',
    #  'whitesmoke', 'yellow', 'yellowgreen']
    # # Human vs Others
    # seq_reps_tsne_emb_human_x, seq_reps_tsne_emb_human_y = [], []
    # seq_reps_tsne_emb_others_x, seq_reps_tsne_emb_others_y = [], []
    # for seq_rep_id in range(seq_reps_tsne_emb.shape[0]):
    #     seq_rep_tsne_emb = seq_reps_tsne_emb[seq_rep_id]
    #     seq_family = NP_tag[seq_rep_id]
    #     if seq_family == 'human':
    #         seq_reps_tsne_emb_human_x.append(seq_rep_tsne_emb[0])
    #         seq_reps_tsne_emb_human_y.append(seq_rep_tsne_emb[1])
    #     else:
    #         seq_reps_tsne_emb_others_x.append(seq_rep_tsne_emb[0])
    #         seq_reps_tsne_emb_others_y.append(seq_rep_tsne_emb[1])
    # plt.scatter(seq_reps_tsne_emb_human_x, seq_reps_tsne_emb_human_y, s=10, color='rosybrown', alpha=0.7, label='Human')
    # plt.scatter(seq_reps_tsne_emb_others_x, seq_reps_tsne_emb_others_y, s=10, color='royalblue', alpha=0.7, label='Others')
    
    # Families
    # fams = list(set(fam_list))
    # fams.sort()
    # legend_handles, fam_handles = [], []
    # for i, fam in enumerate(fam_list):
    #     color = colors[fams.index(fam)]
    #     if fam not in fam_handles:
    #         handle = plt.scatter(seq_reps_tsne_emb[i][0], seq_reps_tsne_emb[i][1], s=10, color=color, alpha=0.7, label=f'{fam}')
    #         legend_handles.append(handle)
    #         fam_handles.append(fam)
    #     else:
    #         plt.scatter(seq_reps_tsne_emb[i][0], seq_reps_tsne_emb[i][1], s=10, color=color, alpha=0.7)
    # plt.xlabel('dim1')
    # plt.ylabel('dim2')
    # plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)
    # plt.savefig(rf'/home/xuyi/rotation1/ESM-tSNE/t-SNE_aug_families.pdf', dpi=300, bbox_inches='tight')
    # print('t-SNE algorithm embedding finished!')
tsne_embedding('1')