from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from tsnecuda import TSNE
import seaborn as sns
import pandas as pd  
import numpy

from parser import parse_args

hyper_params = {}
args = parse_args()

# darkgrid, whitegrid, dark, white and ticks
sns.set_style('ticks')


if args.dataset == "BOOTS":
    aspect_list = "Color|Comfort|Durable|Look|Materials|None|Price|Size|WT_resist"
elif args.dataset == "BAGS_AND_CASES":
    aspect_list = "Compartments|Service|Handles|Looks|None|Price|Protection|Quality|Size_Fit"
elif args.dataset == "TV":
    aspect_list = "Apps|Connectivity|Service|Ease_of_use|Image|None|Price|Size_Look|Sound"
elif args.dataset == "KEYBOARDS":
    aspect_list = "BD_Quality|Connectivity|Extra_function|Feel_Comfort|Layout|Looks|Noise|None|Price"
elif args.dataset == "VACUUMS":
    aspect_list = "Accessories|BD_Quality|Service|Ease_of_use|Noise|None|Price|SC_Power|Weight"
elif args.dataset == "BLUETOOTH":
    aspect_list = "Battery|Comfort|Connectivity|Durable|Ease_of_use|Look|None|Price|Sound"

aspect_lt = aspect_list.split('|')

asp_map = {}
for ind in range(9):
    asp_map[ind] = aspect_lt[ind]

color_label_set = list(sns.color_palette("tab10")) + list(sns.color_palette("Set2")) + list(sns.color_palette("Paired")) + list(sns.color_palette("hls", 80))

def tsne_embedding(sentence_vec, vec_type_set, epoch):
    # sentence_vec = []
    # vec_type_set = {}numpy.array(npsentence_vec)

    # print('numpy.array(sentence_vec) = ', numpy.array(sentence_vec).shape)
    # input()

    embeddings = TSNE(n_components=2, perplexity=10, learning_rate=20).fit_transform(numpy.array(sentence_vec))
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]

    pd_x, pd_y, pd_z, pd_color = [], [], [], []
    color_dict = {}

    for item, value in vec_type_set.items():
        index = list(vec_type_set[item].keys())
        enti_num = len(list(vec_type_set[item].keys()))

        vis_x = embeddings[index, 0]
        vis_y = embeddings[index, 1]

        pd_x.extend(vis_x.tolist())
        pd_y.extend(vis_y.tolist())
        pd_z.extend([aspect_lt[item]] * enti_num)
        pd_color.extend(color_label_set[item])

        color_dict[aspect_lt[item]] = color_label_set[item]

    data_dic = pd.DataFrame({'x': pd_x, 'y': pd_y, 'z': pd_z})

    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("ticks")
    sns.lmplot(x = 'x', y = 'y', data = data_dic, fit_reg=False, legend=True, palette=color_dict , hue='z', size=8, scatter_kws={'s': 50,'alpha':0.6})
    plt.tick_params(labelsize=12)
    plt.savefig(args.pic_file + '/TSNE_type_fig{:d}.png'.format(epoch))
    plt.close()

