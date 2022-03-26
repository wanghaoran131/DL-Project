import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from parser import parse_args

hyper_params = {}
args = parse_args()

# darkgrid, whitegrid, dark, white and ticks
sns.set_style('ticks')

plt.rc('axes', titlesize=18)    # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)   # fontsize of the tick labels
plt.rc('ytick', labelsize=13)   # fontsize of the tick labels
plt.rc('legend', fontsize=13)   # legend fontsize
plt.rc('font', size=13)         # controls default text sizes


def group_data(*data):
    df = pd.concat(data)
    df = df.groupby([df.index, "Aspect", "Name"], as_index=False).sum()
    return df


def plot_figure(predict_true, predict_total, ground_true, epoch):

    df_1 = pd.DataFrame(data={
            "Index": list(range(len(predict_true.keys()))),
            "Aspect": predict_true.keys(),
            "Number": predict_true.values(),
            "Name": "pred truth"})

    df_2 = pd.DataFrame(data={
            "Index": list(range(len(predict_total.keys()))),
            "Aspect": predict_total.keys(),
            "Number": predict_total.values(),
            "Name": "pred false"})


    df_3 = pd.DataFrame(data={
            "Index": list(range(len(ground_true.keys()))),
            "Aspect": ground_true.keys(),
            "Number": ground_true.values(),
            "Name": "ground truth"})

    barplot = group_data(df_1, df_3)

    plt.figure(figsize=(12, 6), tight_layout=True)
    ax = sns.barplot(x=barplot["Index"], y=barplot["Number"], hue=barplot["Name"], palette="pastel")



    ax.set(title="", xlabel="Aspect", ylabel="Number")
    ax.legend(title="Approaches", title_fontsize="13", loc="upper right")

    plt.savefig(args.pic_file + '/plot_fig{:d}.png'.format(epoch))
    plt.close()