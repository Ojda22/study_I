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
    # data = data[data["minimal_mutants"] < 2000]

    correlation_dict = get_correlation(x_variable=data["relevant_mutants"], y_variable=data["mutants_on_change"])
    print("Relevant:Change")
    print(correlation_dict)

    correlation_dict = get_correlation(x_variable=data["minimal_relevant_mutants"],
                                       y_variable=data["mutants_on_change"])
    print("Relevant-Minimal:Change")
    print(correlation_dict)

    correlation_dict = get_correlation(x_variable=data["minimal_relevant_mutants"], y_variable=data["minimal_mutants"])
    print("Relevant-Minimal:Minimal")
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


def agregation_functions(data):
    print(data.describe())

    print(data.groupby(["CommitID", "Hunks"])["Mutants"].sum().groupby('Hunks').sum().values)
    print(data.groupby(["Hunks"]).count().values[:, 1])

    mutants_s = pd.Series(data.groupby(["CommitID", "Hunks"])["Mutants"].sum().groupby('Hunks').sum().values)
    commits_s = pd.Series(data.groupby(["Hunks"]).count().values[:, 1])
    hunks_s = pd.Series(data.groupby(["Hunks"]).count().index.values)
    print()
    frame = {'Mutants': mutants_s, 'Commits': commits_s, "Hunks": hunks_s}

    result = pd.DataFrame(frame)

    result["Average_Mutants_per_Hunk"] = result["Mutants"] / result["Commits"]

    print(result)

    # df = pd.DataFrame(columns=["Mutants", "Commits", "Index"], data=[])

def fault_revelation(data):
    print(data.describe())
    print(data.columns)

    print(data.groupby(["mutant_pool", "percentage", "exists"]).sum())

    mutants_pools = data["mutant_pool"].unique()
    percentages = data["percentage"].unique()

    data_map = []
    for pool in mutants_pools:
        pool_data = data[data["mutant_pool"] == pool]
        for percentage in percentages:
            percentage_pool_data = pool_data[pool_data["percentage"] == percentage]

            does_not = len(percentage_pool_data[percentage_pool_data['exists'] == 0])
            exists = len(percentage_pool_data[percentage_pool_data['exists'] == 1])

            if does_not == 0:
                ms = 1.0
            elif exists == 0:
                ms = 0.0
            else:
                ms = exists / (exists + does_not)

            data_map.append({"pool" : pool, "percentage" : percentage, "ms" : ms})

    print(data_map)

    data = pd.DataFrame(data=data_map)

    print(data)
    print()

def agregation_functions_mutants_operators(data):
    # data.columns = data.columns.to_series().apply(lambda x: x.strip().split("mutators.")[-1])

    data["Mutant_Type"] = data["Mutant_Type"].apply(lambda x: x.strip().split("mutators.")[-1])

    print(data.groupby(["Mutant_Type"])["Mutants"].sum())
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to perform statistics")
    parser.add_argument("-p", "--path_to_data_file",
                        default="/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I",
                        action="store",
                        help="Set path to a data file")

    arguments = parser.parse_args()
    dataframe = pd.read_csv(filepath_or_buffer=arguments.path_to_data_file, thousands=",")
    print()

    # descriptive_information(data=dataframe)
    # calculate_correlation(data=dataframe)
    # agregation_functions(data=dataframe)
    # agregation_functions_mutants_operators(data=dataframe)
    fault_revelation(data=dataframe)
