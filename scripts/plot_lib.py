import os
import sys
import argparse
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(data):

    fig, axes = plt.subplots(ncols=2, nrows=1)

    fig.set_figheight(5)
    fig.set_figwidth(12)

    axeLeft = sns.scatterplot(data=data, x="relevant_mutants", y="mutants_on_change", ax=axes[0])
    axeRight = sns.scatterplot(data=data, x="minimal_mutants", y="minimal_relevant_mutants", ax=axes[1])

    axeRight.set_ylabel("Minimal mutants", fontsize=12)
    axeRight.set_xlabel("Minimal sufficient mutants", fontsize=12)

    axeLeft.set_ylabel("Mutants on a change", fontsize=12)
    axeLeft.set_xlabel("Sufficient mutants", fontsize=12)

    plt.savefig(os.path.join("./plots", "Scatter_plot_correlation.pdf"),
                format='pdf', dpi=1200)

    plt.show()

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Script to perform statistics")
    parser.add_argument("-p", "--path_to_data_file",
                        default="/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I",
                        action="store",
                        help="Set path to a data file")

    arguments = parser.parse_args()
    dataframe = pd.read_csv(filepath_or_buffer=arguments.path_to_data_file, thousands=",")

    # scatter_plot(data=dataframe)

