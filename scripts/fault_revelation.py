import argparse
import random

import pandas as pd
import numpy as np
import os
import sys
sys.path.append("/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I")

from scripts.mutation_comparision import map_mutants, calculate_minimal_mutants


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comparing mutants")
    parser.add_argument("-s", "--statistics_file", action="store", help="File where information about mutants is")
    parser.add_argument("-p", "--path_to_mutants_data", action="store", help="Set path to mutants data")
    parser.add_argument("-o", "--output_dir", action="store", help="Set path to output directory")

    return parser

if __name__ == '__main__':
    arguments = parse_args().parse_args()

    data_frame = pd.read_csv(arguments.statistics_file, index_col="commit_id", thousands=",")

    for _index, row in data_frame.iterrows():

        mutationMatrixPath = arguments.path_to_mutants_data + "/" + row["project_name"] + "/" + _index + "/" + "mutationMatrix.csv"
        mutantsInfoPath = arguments.path_to_mutants_data + "/" + row["project_name"] + "/" + _index + "/" + "mutants_info.csv"

        assert os.path.exists(mutationMatrixPath), "Does not exists: " + mutationMatrixPath
        assert os.path.exists(mutantsInfoPath), "Does not exists: " + mutantsInfoPath

        print(row["tests"])

        faulty_tests = row["tests"]
        faulty_tests = faulty_tests.split(",")
        faulty_tests_names = [test.split("::")[-1] for test in faulty_tests]

        print(_index)
        all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants_no_change = map_mutants(
            mutants_info_path=mutantsInfoPath, mutation_matrix_path=mutationMatrixPath)

        minimal_mutants, subsumed_killed_mutants, mutants_killed, equivalent = calculate_minimal_mutants(
            all_granularity_level)
        subsumed_killed_mutants.update(equivalent)

        minimal_relevant_mutants, minimal_subsumed_killed_mutants, minimal_mutants_killed, minimal_equivalent = calculate_minimal_mutants(
            relevant_mutants)

        print()

        print("All Mutants: {number}".format(number=len(all_fom_mutants)))
        print("All Gran Mutants: {number}".format(number=len(all_granularity_level)))
        print("Relevant: {number}".format(number=len(relevant_mutants)))
        print("Modification: {number}".format(number=len(on_change_mutants)))
        print("Minimal relevant: {number}".format(number=len(minimal_relevant_mutants)))
        print("Minimal: {number}".format(number=len(minimal_mutants)))

        matched_all_m = []
        for mutant in all_granularity_level:
            for test in mutant.killingTests:
                if test.split(".")[-1] in faulty_tests_names:
                    matched_all_m.append(mutant)

        print("All Mutants fault triggering mutants: {number}".format(number=len(matched_all_m)))

        matched_Relevant_m = []
        for mutant in relevant_mutants:
            for test in mutant.killingTests:
                if test.split(".")[-1] in faulty_tests_names:
                    matched_Relevant_m.append(mutant)

        print("Relevant fault triggering mutants: {number}".format(number=len(matched_Relevant_m)))

        matched_mod_m = []
        for mutant in on_change_mutants:
            for test in mutant.killingTests:
                if test.split(".")[-1] in faulty_tests_names:
                    matched_mod_m.append(mutant)

        print("Mod fault triggering mutants: {number}".format(number=len(matched_mod_m)))

        matched_rel_min_m = []
        for mutant in minimal_relevant_mutants:
            for test in mutant.killingTests:
                if test.split(".")[-1] in faulty_tests_names:
                    matched_rel_min_m.append(mutant)

        print("Min relevant fault triggering mutants: {number}".format(number=len(matched_rel_min_m)))

        matched_min_m = []
        for mutant in minimal_mutants:
            for test in mutant.killingTests:
                if test.split(".")[-1] in faulty_tests_names:
                    matched_min_m.append(mutant)

        print("Min fault triggering mutants: {number}".format(number=len(matched_min_m)))
        developer_simulation = arguments.output_dir + "/fault_revelation_msT.csv"
        with open(developer_simulation, "a+") as output_file:
            if os.stat(developer_simulation).st_size == 0:
                output_file.write(
                    "commit,mutant_pool,percentage,iteration,ms\n")

            for pool, name in [(all_fom_mutants, "all"), (relevant_mutants, "relevant"),
                               (on_change_mutants, "modification"), (minimal_relevant_mutants, "minimal_relevant"),
                               (minimal_mutants, "minimal")]:
                for percentage in np.arange(0.01, 1.01, 0.01):

                    for iteration in range(1, 101):
                        to_sample = len(pool) * percentage
                        to_sample_n = int(round(to_sample))
                        sampled_pool = random.sample(pool, to_sample_n)
                        sampled_reveiling = set()
                        for mutant in sampled_pool:
                            for test in mutant.killingTests:
                                if test.split(".")[-1] in faulty_tests_names:
                                    sampled_reveiling.add(mutant)

                        ms = 0
                        if len(sampled_reveiling) != 0:
                            ms = len(sampled_reveiling) / len(sampled_pool)

                        output_file.write("{},{},{},{},{}".format(_index, name, str(round(percentage, 2)), iteration, str(round(ms, 2))))
                        output_file.write("\n")