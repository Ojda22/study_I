import os
import sys
import argparse
import pandas as pd

pd.options.display.float_format = "{:.2f}".format

pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

import scipy.stats.stats as ss

import matplotlib.pyplot as plt
import seaborn as sea


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


def agregation_mutants(data):
    print(data.describe())

    # print(data.groupby(["total_hunks"])["commit"].count())
    grouped_relevant = data.groupby(["total_hunks"])["minimal_relevant_mutants"].sum().reset_index()
    grouped_commits = data.groupby(["total_hunks"])["commit"].count().reset_index()
    grouped_relevant["commits"] = grouped_commits.commit
    merged = pd.DataFrame(grouped_relevant)
    print(merged)
    merged.to_csv("./aggregation_minimal_relevant.csv", index=False, sep=",", header=True)

    # grouped_minimal_relevant = data.groupby(["total_hunks"])["minimal_relevant_mutants"].sum().reset_index()
    # grouped_relevant_df = pd.DataFrame(grouped_relevant[grouped_relevant.total_hunks <= 20])
    # grouped_minimal_relevant_df = pd.DataFrame(grouped_minimal_relevant[grouped_minimal_relevant.total_hunks <= 20])
    #
    # plt.figure(figsize=(12,8))
    #
    # chart = sea.lineplot(x='total_hunks', y='relevant_mutants', data=grouped_relevant_df,
    #                      markers=True)
    # chart.set(ylabel='# Of Commit Relevant Mutants', xlabel="Hunks")
    # import numpy as np
    # z = np.polyfit(grouped_relevant_df.total_hunks, grouped_relevant_df.relevant_mutants, 1)
    # p = np.poly1d(z)
    # plt.plot(grouped_relevant_df.total_hunks, p(grouped_relevant_df.total_hunks), c="b", ls=":")
    #
    # plt.show()
    # print(grouped_minimal_relevant_df)


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

            data_map.append({"pool": pool, "percentage": percentage, "ms": ms})

    print(data_map)

    data = pd.DataFrame(data=data_map)

    print(data)
    print()


def agregation_functions_mutants_operators(data):
    # data.columns = data.columns.to_series().apply(lambda x: x.strip().split("mutators.")[-1])

    data["Mutant_Type"] = data["Mutant_Type"].apply(lambda x: x.strip().split("mutators.")[-1])
    print(data.groupby(["Mutant_Type"])["Mutants"].sum())
    print()
    print("======== Minimal RELEVANT MUTANTS +++++")
    print()
    print(data.groupby(["Mutant_Type"])["Minimal_Mutants"].sum())

    print("======== ALL MUTANTS +++++")
    print()
    print(data.groupby(["Mutant_Type"])["All_Mutants"].sum())

    print()


def features_distribution(dataframe):
    print(len(dataframe))
    dataframe = dataframe.drop_duplicates(subset=["Project", "Mutator", "SourceFile", "MutatedClass", "MutatedMethod", "LineNumber", "Index", "Block", "MethodDescription"], keep='last').reset_index(drop=True)
    print(len(dataframe))

    # distances = ["distanceOfMutantAndPatchInSourceCode", "distanceOfMutantAndPatchInCFG",
    #              "numberOfVariablesUsedInChange_DependsOnMutant", "numberOfVariablesUsedInChange_MutantDependsOn",
    #              "BlockDepth", "CfgDepth", "CfgPredNum", "CfgSuccNum", "NumOfInstructionsInBlock", "NumOutDataDeps",
    #              "NumInDataDeps", "NumOutCtrlDeps", "NumInCtrlDeps", "xAstNumberOfParens", "yAstNumberOfParens",
    #              "xAstNumberOfChildren", "yAstNumberOfChildren"]

    distances = ["CfgDepth", "NumOutDataDeps", "NumInDataDeps", "NumOutCtrlDeps", "NumInCtrlDeps"]

    test_df = pd.DataFrame()

    for feature in distances:
        print(feature)
        df = dataframe[[feature, "Relevant"]]
        # relevantData = df.loc[(df[feature] > 0) & (df['Relevant'] == 1)]
        a = pd.Series((df.loc[(df['Relevant'] == 1)][feature]).to_list(), name=f"R_{feature}")
        # not_relevantData = df.loc[(df[feature] > 0) & (df['Relevant'] == 0)]
        b = pd.Series((df.loc[(df['Relevant'] == 0)][feature]).to_list(), name=f"N_{feature}")
        test_df = pd.concat([test_df, a, b], axis=1).dropna()

        # print("====== RELEVANT ")
        # print()
        # print(relevantData.describe())
        # print("====== NOT RELEVANT ")
        # print()
        # print(not_relevantData.describe())

    # calculate the correlation matrix
    corr = test_df.corr(method="spearman")
    print(corr)

    indexes = [v for v in corr.index.to_list() if v.startswith("N_")]
    columns = [v for v in corr.index.to_list() if v.startswith("R_")]
    # print(corr[str(corr.index).startswith("R")].index)
    corr_extent = corr.drop(index=indexes, columns=columns)

    plt.figure(figsize=(15, 12))

    # Colors
    cmap = sea.diverging_palette(500, 10, as_cmap=True)

    import numpy as np
    # remove the top right triange - duplicate information
    # mask = np.zeros_like(corr, dtype=np.bool)
    # mask[np.triu_indices_from(mask)] = True
    # Getting the Upper Triangle of the co-relation matrix
    # mask = np.triu(corr)

    # plot the heatmap
    ans = sea.heatmap(corr,
                xticklabels=corr.index,
                yticklabels=corr.columns, linewidths=1, cmap=cmap, center=0, annot=True)
    loc, labels = plt.xticks()
    ans.set_xticklabels(labels, rotation=35)
    plt.savefig(os.path.join("/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I/plots",
                             "Heatmap:{}:{}.pdf".format('Mutants', "properties_spearman")),
                format='pdf')
    plt.tight_layout()
    plt.show()

    # for feature in distances:
    #     print()
    #     print(feature)
    #     df = dataframe[[feature, "Relevant", "Minimal"]]
    #     minimal_relevantData = df.loc[(df[feature] > 0) & (df['Relevant'] == 1) & (df['Minimal'] == 1)]
    #     minimal_not_relevantData = df.loc[(df[feature] > 0) & (df['Relevant'] == 0) & (df['Minimal'] == 1)]
    #     print("====== MINIMAL RELEVANT ")
    #     print()
    #     print(minimal_relevantData.describe())
    #     print("====== MINIMAL NOT RELEVANT ")
    #     print()
    #     print(minimal_not_relevantData.describe())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to perform statistics")
    parser.add_argument("-p", "--path_to_data_file",
                        default="/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I",
                        action="store",
                        help="Set path to a data file")

    arguments = parser.parse_args()
    dataframe = pd.read_csv(filepath_or_buffer=arguments.path_to_data_file, thousands=",")
    # print()

    # descriptive_information(data=dataframe)
    # calculate_correlation(data=dataframe)
    # agregation_functions(data=dataframe)
    # agregation_mutants(data=dataframe)
    # agregation_functions_mutants_operators(data=dataframe)
    # fault_revelation(data=dataframe)
    features_distribution(dataframe=dataframe)
    print("test")
