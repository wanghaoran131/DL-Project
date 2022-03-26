import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# # bag
# Compartments|Customer_service|Handles|Looks|None|Price|Protection|Quality|Size_Fit
# # b/t
# Battery|Comfort|Connectivity|Durability|Ease_of_use|Look|None|Price|Sound
# # boots
# Color|Comfort|Durability|Look|Materials|None|Price|Size|Weather_resistance
# # kb
# Build_Quality|Connectivity|Extra_functionality|Feel_Comfort|Layout|Looks|Noise|None|Price
# # TVs
# Apps_Interface|Connectivity|Customer_service|Ease_of_use|Image|None|Price|Size_Look|Sound
# # Vacs
# Accessories|Build_Quality|Customer_Service|Ease_of_use|Noise|None|Price|Suction_Power|Weight

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
elif args.dataset[:4] == "REST":
    aspect_list = "RESTAURANT#GENERAL|FOOD#QUALITY|SERVICE#GENERAL|AMBIENCE#GENERAL|FOOD#STYLE_OPTIONS|FOOD#PRICES|RESTAURANT#MISCELLANEOUS|RESTAURANT#PRICES|DRINKS#QUALITY|DRINKS#STYLE_OPTIONS|LOCATION#GENERAL|DRINKS#PRICES"

aspect_lt = aspect_list.split('|')

asp_map = {}
for ind in range(args.st_num_aspect):
    asp_map[ind] = aspect_lt[ind]


def group_data(*data):
    df = pd.concat(data)
    df = df.groupby([df.index, "Aspect", "Name"], as_index=False).sum()
    return df


def plot_figure(predict_true, predict_total, ground_true, epoch):

    df_1 = pd.DataFrame(data={
            "Index": list(range(len(predict_true.keys()))),
            "Aspect": aspect_lt,
            "Number": list(predict_true.values()),
            "Name": ["pred truth"] * args.st_num_aspect})

    df_2 = pd.DataFrame(data={
            "Index": list(range(len(predict_total.keys()))),
            "Aspect": aspect_lt,
            "Number": predict_total.values(),
            "Name": "pred false"})


    df_3 = pd.DataFrame(data={
            "Index": list(range(len(ground_true.keys()))),
            "Aspect": aspect_lt,
            "Number": ground_true.values(),
            "Name": "ground truth"})


    df_4 = pd.DataFrame(data={
            "Index": list(range(len(ground_true.keys()))),
            "Aspect": aspect_lt,
            "Number": [0] * len(ground_true.keys()),
            "Name": "ground truth"})

    barplot = group_data(df_2, df_3)
    # 42bd79 #cfcf7c #9bdb95  #e8a2ac #a2a4e8
    plt.figure(figsize=(12, 9), tight_layout=True)
    bar1 = sns.barplot(x=barplot["Aspect"], y=barplot["Number"], hue=barplot["Name"], palette=["#c7f2da", "#ff9eab"])
    # "#59ff72"
    barplot = group_data(df_1, df_3)
    # bar2 = sns.barplot(x=barplot["Aspect"], y=barplot["Number"], hue=barplot["Name"], estimator=sum, ci=None, palette=["#55e66b", "#91e4ff"])
    bar2 = sns.barplot(x=barplot["Index"], y=barplot["Number"], hue=barplot["Name"], estimator=sum, ci=None, palette=["#55e66b", "#91e4ff"])

    # top_bar = mpatches.Patch(color='darkblue', label='smoker = No')
    # bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
    # plt.legend(handles=[top_bar, bottom_bar])


    # bar2.set(title="", xlabel="Aspect", ylabel="Number")
    # bar2.legend(title="Approaches", title_fontsize="13", loc="upper right")

    # top_bar = mpatches.Patch(color='darkblue', label='smoker = No')
    # bottom_bar = mpatches.Patch(color='lightblue', label='smoker = Yes')
    
    bar2.set(title="", xlabel="Aspect", ylabel="Number")
    bar2.get_legend().remove()
    # bar2.legend(title="Approaches", title_fontsize="13", loc="upper right")

    plt.savefig(args.pic_file + '/plot_fig{:d}.png'.format(epoch))
    plt.close()