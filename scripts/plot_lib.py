import os
import sys
import argparse
import pandas as pd
from numpy import mean, std

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

def grouped_box_plots_computation_effort_simulation(data, output_dir):
    print("Columns:")
    print(data.columns)
    print()
    print(data["mutant_pool"].unique())
    print(data["target"].unique())

    data = data[data["target"] == "Target_Relevant"]

    data = data.rename(columns={"ms_progression": "MS"})

    data["MS"] = data["MS"].round(0).astype(int)

    mapping_dict = []
    for commit in data.commit.unique():
        commit_data = data.loc[data["commit"] == commit]
        for limit in commit_data.limit.unique():
            limit_data = commit_data.loc[commit_data["limit"] == limit]
            for percentage in limit_data.MS.unique():
                percentage_data = limit_data.loc[limit_data.MS == percentage]

                all = percentage_data[percentage_data["mutant_pool"] == "all"]
                relevant = percentage_data[percentage_data["mutant_pool"] == "relevant"]
                modification = percentage_data[percentage_data["mutant_pool"] == "modification"]
                minimal_relevant = percentage_data[percentage_data["mutant_pool"] == "minimal_relevant"]
                minimal = percentage_data[percentage_data["mutant_pool"] == "minimal"]

                # print(len(all))
                # print(len(relevant))
                # print(len(modification))
                # print(len(predicted))

                for iteration in range(0, np.min([len(all), len(relevant), len(modification), len(minimal_relevant), len(minimal)])):
                    selected_all = all.head(1)
                    all = all.iloc[1:]

                    selected_relevant = relevant.head(1)
                    relevant = relevant.iloc[1:]

                    selected_modification = modification.head(1)
                    modification = modification.iloc[1:]

                    selected_minimal_relevant = minimal_relevant.head(1)
                    minimal_relevant = minimal_relevant.iloc[1:]

                    selected_minimal = minimal.head(1)
                    minimal = minimal.iloc[1:]

                    for technique, mutants_picked, tests_picked in [("All", selected_all["mutants_picked"].iloc[0], selected_all["tests_picked"].iloc[0]),
                                                                    ("Relevant", selected_relevant["mutants_picked"].iloc[0], selected_relevant["tests_picked"].iloc[0]),
                                                                    ("Modification", selected_modification["mutants_picked"].iloc[0], selected_modification["tests_picked"].iloc[0]),
                                                                    ("Minimal_Relevant", selected_minimal_relevant["mutants_picked"].iloc[0], selected_minimal_relevant["tests_picked"].iloc[0]),
                                                                    ("Minimal", selected_minimal["mutants_picked"].iloc[0], selected_minimal["tests_picked"].iloc[0])]:
                        mapping_dict.append({"Technique": technique, "Mutants_Picked": mutants_picked, "Tests_picked": tests_picked, "MS": percentage})


    print()

    data = pd.DataFrame(mapping_dict)

    print(data.columns)

    data['Technique'] = data['Technique'].replace({ "All" : "Random" })

    # data = data.loc[data['Technique'].isin(["Random", "Predicted", "Minimal", "Relevant"])]
    # data = data.loc[data['Technique'].isin(["Random", "Minimal", "Relevant", "Minimal_Relevant"])]
    data = data.loc[data['Technique'].isin(["Random", "Relevant", "Minimal_Relevant"])]

    stats = data.describe(include='all').loc[["mean", "std"]]
    print(stats)

    # calculate summary statistics
    data_mean, data_std = mean(data["Tests_picked"]), std(data["Tests_picked"])
    print("Mean:" , data_mean)
    print("Std:" , data_std)
    # identify outliers
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    print("Lower: ", lower)
    print("Upper: ", upper)
    q = data["Tests_picked"].quantile(0.99)
    print("Upper quantile: ", q)
    min, max = np.min(data["Tests_picked"]), np.max(data["Tests_picked"])
    print("Min: " , min)
    print("Max: " , max)
    # identify outliers
    outliers = [x for x in data["Tests_picked"] if x < lower or x > upper]
    print('Identified outliers: ')
    print("Number: ", len(outliers))
    # print("Range: {}-{}".format(np.min(outliers), np.max(outliers)))

    print(data.dtypes)
    data = data.loc[data["Tests_picked"] < 2500]
    print("After")
    min, max = np.min(data["Tests_picked"]), np.max(data["Tests_picked"])
    print("Min: ", min)
    print("Max: ", max)

    print(data['Technique'].unique())

    # Share both X and Y axes with all subplots
    # fig, axes = plt.subplots(ncols=2, nrows=1, sharex='all', sharey='all')
    fig, axes = plt.subplots(ncols=1, nrows=1)
    fig.set_figheight(12)
    fig.set_figwidth(10)
    # fig.add_subplot(111, frameon=False)

    my_pal = {'Random': 'cornflowerblue', 'Relevant': 'palegreen', 'Minimal_Relevant': 'turquoise'}
    # my_pal = {'Random': 'cornflowerblue', 'Relevant': 'palegreen', "Minimal": "firebrick"}

    # m_picked = sea.boxplot(x="Technique", y="Mutants_Picked", data=data, ax=axes[0], palette=my_pal)
    # # m_picked = sea.swarmplot(x="Technique", y="Mutants_Picked", data=data, color=".25")
    # m_picked.set_ylabel("Number of mutants", fontsize=18)
    # m_picked.tick_params(labelsize=12, axis='x')
    # m_picked.tick_params(labelsize=22, axis='y')
    # m_picked.set_title("Human effort")
    # m_picked.set_xlabel("", fontsize=12)

    t_picked = sns.boxplot(x="Technique", y="Tests_picked", data=data, ax=axes, palette=my_pal)
    t_picked.set_ylabel("Number of tests", fontsize=18)
    t_picked.tick_params(labelsize=15, axis='x')
    t_picked.tick_params(labelsize=22, axis='y')
    t_picked.set_title("Computation effort")
    t_picked.set_xlabel("", fontsize=15)
    # t_picked.set_yticks(np.arange(0, max, 100))

    # t_picked.margins(x=0.0)

    print("Minumum value: ", str(np.min(data["Tests_picked"])))

    #     hatches = ['/', '*', 'x', "O", "*"]
    hatches = ['/', '*', '.']
    # hatches = ['/', '*', "O"]
    i = 0
    # for picked_bar, tests_picked in zip(m_picked.patches, t_picked.patches):
    for tests_picked in t_picked.patches:
        if i == 3:
            i = 0
        # picked_bar.set_hatch(hatches[i])
        tests_picked.set_hatch(hatches[i])
        i += 1

    i = 0
    # for picked_bar, tests_picked in zip(m_picked.artists, t_picked.artists):
    for tests_picked in t_picked.artists:
        if i == 3:
            i = 0
        # picked_bar.set_hatch(hatches[i])
        tests_picked.set_hatch(hatches[i])
        i += 1

    # plt.xlabel("Techniques")
    # plt.ylabel("common Y")


    # fig.text(0.55, 0.01, 'Selections', ha='center', va='center')
    # fig.text(0.01, 0.5, 'common ylabel', ha='center', va='center', rotation='vertical')

    plt.savefig(os.path.join(output_dir, "Box_plot:{}:{}.pdf".format('Simulation', "computational_effort")),
                                             format='pdf',
                                             dpi=1500)
    # plt.savefig(os.path.join(dir_path,"{}_{}.eps".format('Boxplot', "human_effort_v2")),
    #                                          format='eps',
    #                                          dpi=1200)

    # plt.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, wspace=0.2, hspace=0.4)
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

    data = data[data["target"] == "Target_Relevant"]

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


def lineplot(data, output_dir):
    sns.lineplot(data=data, x="percentage", y="ms", hue="mutant_pool")
    plt.savefig(os.path.join(output_dir,
                             "Line_plot:{}:{}.pdf".format('Simulation', "fault_revelation_v2")),
                format='pdf',
                dpi=1200)
    plt.tight_layout()
    plt.show()


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
    # grouped_box_plots_developer_simulation(data=dataframe, output_dir=arguments.output_dir)
    # grouped_box_plots_computation_effort_simulation(data=dataframe, output_dir=arguments.output_dir)
    lineplot(data=dataframe, output_dir=arguments.output_dir)
