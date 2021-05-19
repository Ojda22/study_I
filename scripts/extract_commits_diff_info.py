import argparse
import csv

from pydriller import RepositoryMining
from pydriller.metrics.process.hunks_count import HunksCount
import pandas as pd


import sys
sys.path.append("/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I")

import os
from operator import itemgetter
from itertools import groupby

# import logging
# logging.basicConfig(level=logging.INFO)
from scripts.mutation_comparision import map_mutants


def parse_commit_diff(project_git_url, commit_id):
    # logging.info("<<< Project git url: {path}".format(path=project_git_url))
    print("<<< Project git url: {path}".format(path=project_git_url))
    print("<<< Commit id: {commit}".format(commit=commit_id))

    modified_files = list()

    # changed_files = dict()
    num_of_files = 0
    total_complexity = 0
    total_added_lines = 0
    total_hunks = 0
    total_loc = 0
    added_lines = dict()
    dmm_unit_complexity = 0
    dmm_unit_size = 0
    dmm_unit_interfacing = 0

    for commit in RepositoryMining(path_to_repo=project_git_url, single=commit_id).traverse_commits():
        num_of_files = len(commit.modifications)
        dmm_unit_complexity = commit.dmm_unit_complexity
        dmm_unit_size = commit.dmm_unit_size
        dmm_unit_interfacing = commit.dmm_unit_interfacing
        print("<<< Number of files: {num}".format(num=str(num_of_files)))
        for m in commit.modifications:
            total_complexity += m.complexity
            total_added_lines += m.added
            total_loc += m.nloc
            # changed_files[m.filename] = {"file_complexity":m.complexity, "loc":m.nloc, "added_lines":m.added, "tokens":m.token_count}

            modified_files.append(m.filename)
            added_lines[m.filename] = [element[0] for element in m.diff_parsed["added"]]
            print("<<< File name modified: {file}".format(file=m.filename))
            print("<<< Modification new path: {path}".format(path=m.new_path))

    hunks = HunksCount(path_to_repo=project_git_url, from_commit=commit_id, to_commit=commit_id)
    hunks_dict = hunks.count()

    for file, hunks_number in hunks_dict.items():
        total_hunks += hunks_number
    #     file = file.split("/")[-1]
    #     if (file in changed_files.keys()):
    #         changed_files[file].update({"hunks": hunks_number})
    #
    # changed_files["change_overall"] = {"num_of_files": num_of_files, "total_added_lines": total_added_lines,
    #                                    "total_complexity": total_complexity, "total_hunks": total_hunks,
    #                                    "total_loc": total_loc}

    # hunks = HunksCount(path_to_repo=project_git_url, from_commit=commit_id, to_commit=commit_id)
    # hunks_dict = hunks.count()
    #
    # for file, hunks_number in hunks_dict.items():
    #     total_hunks += hunks_number
    #
    # avg_added_lines = total_added_lines / num_of_files
    # avg_complexity = total_complexity / num_of_files
    # avg_hunks = total_hunks / num_of_files
    # avg_loc = total_loc / num_of_files

    # metric = LinesCount(path_to_repo=project_git_url, from_commit=commit_id, to_commit=commit_id)
    # added_count = metric.count()
    # added_max = metric.max_added()
    # added_avg = metric.avg_added()
    # print('Total lines added per file: {}'.format(added_count))
    # print('Maximum lines added per file: {}'.format(added_max))
    # print('Average lines added per file: {}'.format(added_avg))

    with open("./commits_diff_info_stats.csv", "a+") as output_file:
        if os.stat("./commits_diff_info_stats.csv").st_size == 0:
            output_file.write(
                "commit_id,num_of_files,total_loc,total_complexity,total_added_lines,total_hunks,"
                "dmm_unit_complexity,dmm_unit_size,dmm_unit_interfacing\n")

        output_file.write("{commit_id},{num_of_files},{total_loc},{total_complexity},{total_added_lines},"
                          "{total_hunks},{dmm_unit_complexity},"
                          "{dmm_unit_size},{dmm_unit_interfacing}".format(commit_id=commit_id,
                                                                          num_of_files=str(num_of_files),
                                                                          total_loc=str(total_loc),
                                                                          total_complexity=str(total_complexity),
                                                                          total_added_lines=str(total_added_lines),
                                                                          total_hunks=str(total_hunks),
                                                                          dmm_unit_complexity=str(dmm_unit_complexity),
                                                                          dmm_unit_size=str(dmm_unit_size),
                                                                          dmm_unit_interfacing=str(
                                                                              dmm_unit_interfacing)))
        output_file.write("\n")

    print("{commit_id},{num_of_files},{total_loc},{total_complexity},{total_added_lines},{total_hunks},"
          "{dmm_unit_complexity},{dmm_unit_size},{dmm_unit_interfacing}".format(commit_id=commit_id,
                                                                                num_of_files=str(num_of_files),
                                                                                total_loc=str(total_loc),
                                                                                total_complexity=str(total_complexity),
                                                                                total_added_lines=str(
                                                                                    total_added_lines),
                                                                                total_hunks=str(total_hunks),
                                                                                dmm_unit_complexity=str(
                                                                                    dmm_unit_complexity),
                                                                                dmm_unit_size=str(dmm_unit_size),
                                                                                dmm_unit_interfacing=str(
                                                                                    dmm_unit_interfacing)))


def commit_mutants_type(data_file_path, path_to_mutants_data, output_path):
    data_frame = pd.read_csv(data_file_path, index_col="commit", thousands=",")
    mutants_types = set()
    mutants_types_dict = []

    for _index, row in data_frame.iterrows():
        mutationMatrixPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutationMatrix.csv"
        mutantsInfoPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutants_info.csv"

        assert os.path.exists(mutationMatrixPath), "Does not exists: " + mutationMatrixPath
        assert os.path.exists(mutantsInfoPath), "Does not exists: " + mutantsInfoPath

        print(_index)
        all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants = map_mutants(
            mutants_info_path=mutantsInfoPath, mutation_matrix_path=mutationMatrixPath)

        # group relevant mutants based on a mutant type
        [mutants_types.add(mutant_type.mutant_operator) for mutant_type in relevant_mutants]
        # print(mutants_types)
        for mutants_type in mutants_types:
            mutants_types_dict.append({"Mutant_Type": mutants_type,
                                       "Mutants": len(
                [mutant for mutant in relevant_mutants if mutant.mutant_operator == mutants_type]),
                                       "Minimal_Mutants" : len(
                [mutant for mutant in minimal_relevant_mutants if mutant.mutant_operator == mutants_type]),
                                       "All_Mutants" : len(
                [mutant for mutant in all_granularity_level if mutant.mutant_operator == mutants_type])})

    try:
        with open(output_path + "/mutants_type_distribution_extended_v2.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Mutant_Type", "Mutants", "Minimal_Mutants", "All_Mutants"])
            writer.writeheader()
            for data in mutants_types_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")



def commit_mutants_propertires(data_file_path, path_to_mutants_data, output_path):
    data_frame = pd.read_csv(data_file_path, index_col="commit", thousands=",")

    mutants_types = set()
    mutants_types_dict = []

    mutants_on_method = {"In_Method": 0, "Not_In_Method": 0}

    mutants_on_hunks = []
    for _index, row in data_frame.iterrows():
        mutationMatrixPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutationMatrix.csv"
        mutantsInfoPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutants_info.csv"

        assert os.path.exists(mutationMatrixPath), "Does not exists: " + mutationMatrixPath
        assert os.path.exists(mutantsInfoPath), "Does not exists: " + mutantsInfoPath

        print(_index)
        all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants = map_mutants(
            mutants_info_path=mutantsInfoPath, mutation_matrix_path=mutationMatrixPath)

        # group mutants based on a mutant type
        [mutants_types.add(mutant_type.mutant_operator) for mutant_type in relevant_mutants]
        # print(mutants_types)
        for mutants_type in mutants_types:
            mutants_types_dict.append({"Mutant_Type" : mutants_type, "Mutants" : len([mutant for mutant in relevant_mutants if mutant.mutant_operator == mutants_type])})

        for commit in RepositoryMining(path_to_repo="https://github.com/apache/commons-" + row["project"].split("-")[1] + ".git", single=_index).traverse_commits():
            for m in commit.modifications:
                # check if mutants are on changed methods
                if "Test" in m.filename.split("/")[-1]:
                    continue
                changed_methods = []
                for method in m.changed_methods:
                    changed_methods.append(method.name.split("::")[-1])
                mutants_in_file_modified= [mutant for mutant in relevant_mutants if mutant.sourceFile == m.filename.split("/")[-1]]
                for mutant in mutants_in_file_modified:
                    if mutant.mutatedMethod in changed_methods:
                        mutants_on_method["In_Method"] += 1
                    else:
                        mutants_on_method["Not_In_Method"] += 1

                # change_lines_per_file = list(map(lambda x: x[0], m.diff_parsed["added"]))
                # grouped = [map(itemgetter(1), g) for k, g in groupby(enumerate(change_lines_per_file), lambda i_x: i_x[0] - i_x[1])]

        # check mutants associated with hunks
        hunks = HunksCount(path_to_repo="https://github.com/apache/commons-" + row["project"].split("-")[1] + ".git", from_commit=_index, to_commit=_index)
        hunks_dict = hunks.count()

        for file, hunks_number in hunks_dict.items():
            file = file.split("/")[-1]
            if "Test" in file:
                continue
            mutants = len([mutant for mutant in relevant_mutants if mutant.sourceFile == file])
            if mutants == 0:
                print()
            mutants_on_hunks.append({"CommitID" : _index,"Hunks" : hunks_number, "Mutants" : mutants})

    try:
        with open(output_path + "/mutants_type_distribution.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Mutant_Type", "Mutants"])
            writer.writeheader()
            for data in mutants_types_dict:
                writer.writerow(data)
    except IOError:
        print("I/O error")

    with open(output_path + "/mutants_in_changed_method.txt", "w") as output_file:
        output_file.write("Mutant In Method: {} \nMutants Outside Method: {}".format(str(mutants_on_method["In_Method"]),str(mutants_on_method["Not_In_Method"])))

    try:
        with open(output_path + "/mutants_on_hunks.csv", 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["CommitID", "Hunks", "Mutants"])
            writer.writeheader()
            for data in mutants_on_hunks:
                writer.writerow(data)
    except IOError:
        print("I/O error")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract metrices related to the particular commit changes")
    parser.add_argument("-p", "--project_git_url", action="store", help="Target project git url")
    parser.add_argument("-c", "--commit_id", action="store", help="Commit on which to perform analysis")
    parser.add_argument("-s", "--statistics_file", action="store", help="File where information about mutants is")
    parser.add_argument("-m", "--path_to_mutants_data", action="store", help="Set path to mutants data")
    parser.add_argument("-o", "--output_dir", action="store", help="Set path to output directory")

    arguments = parser.parse_args()

    # parse_commit_diff(arguments.project_git_url, arguments.commit_id)

    # commit_mutants_propertires(data_file_path=arguments.statistics_file, path_to_mutants_data=arguments.path_to_mutants_data,
    #                      output_path=arguments.output_dir)
    commit_mutants_type(data_file_path=arguments.statistics_file,
                               path_to_mutants_data=arguments.path_to_mutants_data,
                               output_path=arguments.output_dir)