import os
import sys
import argparse
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

import scipy.stats.stats as ss

def descriptive_information(data):
    data.describe()
    print("Columns: \n", data.columns)
    print("Data type: \n", data.dtypes)


def calculate_correlation(data):
    correlation_dict = get_correlation(x_variable=data["minimal_mutants"], y_variable=data["minimal_relevant_mutants"])
    print(correlation_dict)


def get_correlation(x_variable, y_variable):
    assert (len(x_variable) == len(y_variable)), "X and Y must have same length"
    assert len(x_variable) > 1, "Both X and Y must have at least 2 elements"

    correlation = {}
    cc, p_value = ss.pearsonr(x_variable, y_variable)
    correlation['pearson'] = {'corr': cc, 'p-value': p_value}
    cc, p_value = ss.kendalltau(x_variable, y_variable)
    correlation['kendall'] = {'corr': cc, 'p-value': p_value}
    return correlation


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="Script to perform statistics")
    parser.add_argument("-p", "--path_to_data_file",
                        default="/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I",
                        action="store",
                        help="Set path to a data file")

    arguments = parser.parse_args()
    dataframe = pd.read_csv(filepath_or_buffer=arguments.path_to_data_file, thousands=",")

    descriptive_information(data=dataframe)
    calculate_correlation(data=dataframe)




