import os
import sys
import argparse
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def scatter_plot(data):

    fig, axes = plt.subplots(ncols=3, nrows=1, constrained_layout=True)

    fig.set_figheight(5)
    fig.set_figwidth(12)

    data = data[data["minimal_mutants"] < 2000]

    axeLeft = sns.scatterplot(data=data, x="relevant_mutants", y="mutants_on_change", ax=axes[0])
    axeMiddle = sns.scatterplot(data=data, x="minimal_relevant_mutants", y="mutants_on_change", ax=axes[1])
    axeRight = sns.scatterplot(data=data, x="minimal_relevant_mutants", y="minimal_mutants", ax=axes[2])

    axeLeft.set_ylabel("Mutants on a change", fontsize=12)
    axeLeft.set_xlabel("Relevant Mutants", fontsize=12)

    axeMiddle.set_ylabel("Mutants on a change", fontsize=12)
    axeMiddle.set_xlabel("Minimal Relevant Mutants", fontsize=12)

    axeRight.set_ylabel("Minimal Mutants", fontsize=12)
    axeRight.set_xlabel("Minimal Relevant Mutants", fontsize=12)

    plt.savefig(os.path.join("./plots", "Scatter_plot:correlations.pdf"),
                format='pdf', dpi=1500)

    # plt.tight_layout()
    plt.show()
    print()


def box_plot_stacked(data, output_dir):
    print(data.columns)
    print(data.dtypes)
    print(data.describe())

    data.index = data.index + 1

    # data["size_ratio"] = data['size_ratio'] * 100

    data = data.sort_values(by=["mutants_gran"], ascending=True)

    data["total_mutants"] = data["mutants_gran"]
    data["ratio_of_change"] = data["mutants_on_change"] / data["total_mutants"]
    data["ration_of_relevant"] = data['relevant_mutants'] / data["total_mutants"]
    data["not_relevant"] = 1 - data["ratio_of_change"] - data["ration_of_relevant"]

    # STACKED BARS

    fig, axes = plt.subplots(nrows=1, ncols=1)

    fig.set_figheight(5)
    fig.set_figwidth(12)

    b_top = axes.bar(x=data.index, height=data['ratio_of_change']+data['ration_of_relevant']+data["not_relevant"], facecolor='lightblue', alpha=0.8, label='Ratio of not relevant mutants')
    b_middle = axes.bar(x=data.index, height=data['ratio_of_change']+data['ration_of_relevant'], facecolor='#ea4335', alpha=0.8, label='Ratio of change size')
    b_bottom = axes.bar(x=data.index, height=data['ration_of_relevant'], facecolor='palegreen', alpha=0.8, label='Ratio of relevant mutants')

    # pattern parameter-> , hatch = 'x'

    axes.set_ylabel('Percentages - %')
    axes.set_xlabel('Commits')
    axes.legend(loc='upper right')

    plt.xlim(0, len(data.index)+ 1)
    plt.xticks(np.arange(np.min(data.index), np.max(data.index), 5), fontsize=5, rotation=45)
    plt.yticks(np.arange(0,1, 0.1))

    plt.savefig(os.path.join(output_dir, "Box_plot:Distribution_test2.pdf"),
                format='pdf', dpi=1200)
    plt.tight_layout()
    plt.show()
    print()

def grouped_box_plots_developer_simulation(data, output_dir):
    # data.columns = data.columns.to_series().apply(lambda x: x.strip())

    print("Columns")
    print(data.columns)
    print()
    print(data["mutant_pool"].unique())
    print(data["target"].unique())

    data = data[data["target"] == "Target_Minimal_Relevant"]

    data = data.rename(columns={"ms_progression": "MS"})

    data.mutant_pool = data.mutant_pool.replace(
        {"all": "Random", "relevant": "Relevant", "modification": "Modification", "minimal_relevant": "Minimal Relevant"})
    # data.mutant_pool = data.mutant_pool.replace({"all": "Random", "relevant": "Relevant", "modification": "Modification"})

    # data = data.loc[data.mutant_pool.isin(["Random", "Modification", "Relevant"])]
    data = data.loc[data.mutant_pool.isin(["Random", "Modification", "Minimal Relevant"])]
    print(data.mutant_pool.unique())

    commits = data.commit.unique()

    my_pal = {'Random': 'cornflowerblue', 'Minimal Relevant': 'firebrick', 'Modification': 'pink'}
    # my_pal = {'Random': 'cornflowerblue', 'Relevant': 'palegreen', 'Modification': 'pink', "Minimal Relevant": "deepskyblue"}
    # for commit in commits:
    #     df_commit = data[data["commit"] == commit]

    fig = plt.figure(figsize=(15, 8))
    b = sns.boxplot(x="limit", y="MS", hue="mutant_pool", data=data, palette=my_pal)
    # ax = sns.swarmplot(x="limit", y="MS", data=df_commits, color=".25")
    # hatches = ['/', '+', '//', '-', 'x', '\\', '*', 'o', 'O']
    # hatches = ['/', '*', 'x', "O"]
    # hatches = ['/', '*', "x"]
    hatches = ['/',"x", '*',]
    i = 0
    for bar in b.patches:
        if i == 3:
            i = 0
        bar.set_hatch(hatches[i])
        i += 1

    i = 0
    for patch in b.artists:
        if i == 3:
            i = 0
        patch.set_hatch(hatches[i])
        i += 1

    b.set_alpha(0.5)
    b.set_xlabel("Number of selected mutants", fontsize=22)
    b.set_ylabel("Killed Ratio of Mutants - %", fontsize=22)
    b.tick_params(labelsize=22)
    b.set_title("Progression of Minimal Relevant Mutation Score", fontsize=26)
    # b.set_title(commit)
    b.legend(loc="lower right", fontsize=26)
    plt.savefig(os.path.join(output_dir,
                             "Box_plot:{}:{}.pdf".format('Simulation', "minimal_relevant_ms")),
                format='pdf',
                dpi=1200)
    # plt.savefig(os.path.join(dir_path,
    #                          "{}_{}.eps".format('Boxplot', "progression_MS_prediction_v2")),
    #             format='eps',
    #             dpi=1200)
    plt.tight_layout()
    plt.show()
    print()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Script to perform statistics")
    parser.add_argument("-p", "--path_to_data_file",
                        default="/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I",
                        action="store",
                        help="Set path to a data file")
    parser.add_argument("-o", "--output_dir", action="store", help="Set path to output directory")

    arguments = parser.parse_args()
    dataframe = pd.read_csv(filepath_or_buffer=arguments.path_to_data_file, thousands=",")

    # scatter_plot(data=dataframe)
    # box_plot_stacked(data=dataframe, output_dir=arguments.output_dir)
    grouped_box_plots_developer_simulation(data=dataframe, output_dir=arguments.output_dir)
