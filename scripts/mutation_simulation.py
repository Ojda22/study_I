import argparse
import pandas as pd
import os
import random

import sys
sys.path.append("/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I")

from scripts.mutation_comparision import map_mutants, calculate_minimal_mutants


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulation on mutants")
    parser.add_argument("-s", "--statistics_file", action="store", help="File where information about mutants is")
    parser.add_argument("-p", "--path_to_mutants_data", action="store", help="Set path to mutants data")
    parser.add_argument("-o", "--output_dir", action="store", help="Set path to output directory")

    return parser


def venn_diagram_simulation(data, path_to_mutants_data, output_path):
    for _index, row in data.iterrows():
        mutationMatrixPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutationMatrix.csv"
        mutantsInfoPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutants_info.csv"

        assert os.path.exists(mutationMatrixPath), "Does not exists: " + mutationMatrixPath
        assert os.path.exists(mutantsInfoPath), "Does not exists: " + mutantsInfoPath

        print(_index)
        all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants_no_change = map_mutants(
            mutants_info_path=mutantsInfoPath, mutation_matrix_path=mutationMatrixPath)

        minimal_mutants, subsumed_killed_mutants, mutants_killed, equivalent = calculate_minimal_mutants(all_granularity_level)
        subsumed_killed_mutants.update(equivalent)

        relevant_mutants_with_change = relevant_mutants + on_change_mutants
        minimal_relevant_mutants, minimal_subsumed_killed_mutants, minimal_mutants_killed, minimal_equivalent = calculate_minimal_mutants(
            relevant_mutants_with_change)
        subsumed_killed_mutants.update(equivalent)

        print("Number of subsuming mutants: ", len(minimal_mutants))
        print("Number of subsuming relevant mutants: ", len(minimal_relevant_mutants))
        print("Number of mutants on change: ", len(relevant_mutants))

        hard_to_kill = [mutant for mutant in all_granularity_level if mutant.hard_to_kill_score() <= 0.025]
        print("Number of hard to kill mutants: ", len(hard_to_kill))

        argument_output_file = output_path + "/comparison_qmi.csv"
        with open(argument_output_file, "a+") as output_file:
            if os.stat(argument_output_file).st_size == 0:
                output_file.write("{},{},{},{},{}".format("mid", "subsuming", "relevant_subsuming", "change", "hard_to_kill"))
                output_file.write("\n")

            for m in all_granularity_level:
                subsuming = "1" if m in minimal_mutants else "0"
                relevant_subsuming = "1" if m in minimal_relevant_mutants else "0"
                change = "1" if m in on_change_mutants else "0"
                hard_to_kill = "1" if m.hard_to_kill_score() <= 0.10 else "0"
                output_file.write(
                    "{mid},{subsuming},{relevant_subsuming},{change},{hard_to_kill}".format(mid=str(str(m.mutant_ID) + str(_index)), subsuming=subsuming,
                                                                 relevant_subsuming=relevant_subsuming, change=change, hard_to_kill=hard_to_kill))
                output_file.write("\n")


        # # print("All fom mutants: {number}".format(number=len(all_fom_mutants)))
        # # print("All granularity: {number}".format(number=len(all_granularity_level)))
        # # print("Relevant mutants: {number}".format(number=len(relevant_mutants)))
        # # print("Not relevant mutants: {number}".format(number=len(not_relevant_mutants)))
        # # print("On change mutants: {number}".format(number=len(on_change_mutants)))
        # # print("Minimal relevant mutants: {number}".format(number=len(minimal_relevant_mutants)))
        # # print("Minimal mutants: {number}".format(number=len(minimal_mutants)))
        #
        # # print("\n")
        #
        # minimal_change_intersection = set(minimal_mutants).intersection(set(on_change_mutants))
        # minimal_relevantminimal_intersection = set(minimal_mutants).intersection(set(minimal_relevant_mutants))
        # change_relevantminimal_intersection = set(on_change_mutants).intersection(minimal_relevant_mutants)
        #
        # # minimal_relevant_minimal_change_intersection = minimal_change_intersection.intersection(minimal_relevantminimal_intersection).intersection(change_relevantminimal_intersection)
        # d = [on_change_mutants, minimal_relevant_mutants, minimal_mutants]
        # minimal_relevant_minimal_change_intersection = set.intersection(*[set(x) for x in d])
        #
        # print(len(minimal_change_intersection))
        # print(len(minimal_relevantminimal_intersection))
        # print(len(change_relevantminimal_intersection))
        # print(len(minimal_relevant_minimal_change_intersection))
        #
        # # percentages for minimal
        # percent_minimal_in_change = len(minimal_change_intersection) / len(minimal_mutants)
        # percent_minimal_in_relevant_minimal = len(minimal_relevantminimal_intersection) / len(minimal_mutants)
        # percent_minimal_in_all = len(minimal_relevant_minimal_change_intersection) / len(minimal_mutants)
        #
        # percent_minimal_rest = len([mutant for mutant in minimal_mutants if mutant not in minimal_change_intersection and mutant not in minimal_relevantminimal_intersection and mutant not in minimal_relevant_minimal_change_intersection]) / len(minimal_mutants)
        #
        # # print(percent_minimal_in_change)
        # # print(percent_minimal_in_relevant_minimal)
        # # print(percent_minimal_in_all)
        # # print(percent_minimal_rest)
        #
        # # percentages for change
        # percent_change_in_minimal = len(minimal_change_intersection) / len(on_change_mutants)
        # percent_change_in_relevant_minimal = len(change_relevantminimal_intersection) / len(on_change_mutants)
        # percent_change_in_all = len(minimal_relevant_minimal_change_intersection) / len(on_change_mutants)
        #
        # mutants_just_on_change = len([mutant for mutant in on_change_mutants if mutant not in minimal_change_intersection and mutant not in change_relevantminimal_intersection and mutant not in minimal_relevant_minimal_change_intersection])
        # percent_change_rest = mutants_just_on_change / len(on_change_mutants)
        #
        # # print(percent_change_in_minimal)
        # # print(percent_change_in_relevant_minimal)
        # # print(percent_change_in_all)
        # # print(percent_change_rest)
        #
        # # percentages for minimalRelevant
        # percent_relevantminimal_in_minimal = len(minimal_relevantminimal_intersection) / len(minimal_relevant_mutants)
        # percent_relevantminimal_in_change = len(change_relevantminimal_intersection) / len(minimal_relevant_mutants)
        # percent_relevantminimal_in_all = len(minimal_relevant_minimal_change_intersection) / len(minimal_relevant_mutants)
        #
        # percent_relevantminimal_rest = len([mutant for mutant in minimal_relevant_mutants if mutant not in minimal_relevantminimal_intersection and mutant not in change_relevantminimal_intersection and mutant not in minimal_relevant_minimal_change_intersection]) / len(minimal_relevant_mutants)
        #
        # # print(percent_relevantminimal_in_minimal)
        # # print(percent_relevantminimal_in_change)
        # # print(percent_relevantminimal_in_all)
        # # print(percent_relevantminimal_rest)
        #
        # venn_file_output = output_path + "/venn_diagram_data.csv"
        # with open(venn_file_output, "a+") as output_file:
        #     if os.stat(venn_file_output).st_size == 0:
        #         output_file.write(
        #             "commit,all_granularity,relevant_mutants,change_mutants,minimal_relevant,minimal_mutants,"
        #             "minimal_change_intersection,minimal_relevant-minimal_intersection,change_relevant-minimal_intersection,minimal_relevant-minimal_change_intersection,"
        #             "percent_minimal_in_change,percent_minimal_in_relevant_minimal,percent_minimal_in_all,percent_minimal_rest,"
        #             "percent_change_in_minimal,percent_change_in_relevant_minimal,percent_change_in_all,percent_change_rest,"
        #             "percent_relevant-minimal_in_minimal,percent_relevant-minimal_in_change,percent_relevant-minimal_in_all,percent_relevant-minimal_rest\n")
        #
        #     output_file.write("{},{},{},{},{},{},{},{},{},{},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f}".format(
        #         _index, len(all_granularity_level), len(relevant_mutants), len(on_change_mutants), len(minimal_relevant_mutants), len(minimal_mutants),
        #         len(minimal_change_intersection), len(minimal_relevantminimal_intersection),len(change_relevantminimal_intersection),len(minimal_relevant_minimal_change_intersection),
        #                                                                 percent_minimal_in_change,percent_minimal_in_relevant_minimal,percent_minimal_in_all,percent_minimal_rest,
        #                                                                 percent_change_in_minimal,percent_change_in_relevant_minimal,percent_change_in_all,percent_change_rest,
        #                                                                 percent_relevantminimal_in_minimal,percent_relevantminimal_in_change,percent_relevantminimal_in_all, percent_relevantminimal_rest))
        #
        #
        #
        #     output_file.write("\n")


def write_ms_achieved(output_file, target, commit, mutant_pool, iteration, ms_progression, mutants_picked, limit, chosen_tests):
    output_file.write("{},{},{},{},{},{},{},{}".format(commit, target, mutant_pool, limit, iteration, mutants_picked, chosen_tests,
                                                    ms_progression))
    output_file.write("\n")


def mutation_score_with_tests(list_of_mutants, tests):
    survived = 0
    killed = 0
    for mutant in list_of_mutants:
        killing_tests = mutant.killingTests
        if len(killing_tests.intersection(tests)) > 0:
            killed += 1
        else:
            survived += 1
    if survived == 0 and killed == 0:
        return "No mutants"
    else:
        return round(killed / (survived + killed) * 100, 2)


def get_ms_from_simulation(mutants, relevant_mutants, test_suite, limit):
    # copy the given mutant list
    mutant_pool = list(mutants)
    chosen_tests = list()
    # with 0 mutants the relevant MS is 0
    ms_progression = [0]
    # max_ms should be 100 but not always (in case of mutants with timeout)
    max_ms = mutation_score_with_tests(relevant_mutants, test_suite)
    mutants_picked = 0
    to_run_tests = 0
    while len(mutant_pool) > 0 and ms_progression[-1] != max_ms and mutants_picked < limit:
        mutants_picked = mutants_picked + 1
        # pick a mutant
        chosen_mutant = random.choice(mutant_pool)
        # take it out of the pool of mutants
        mutant_pool.remove(chosen_mutant)
        # if the mutant is killed, add its test to the chosen tests and remove the mutants it kills from the pool
        killing_tests_chosen = chosen_mutant.killingTests
        if len(killing_tests_chosen) > 0:
            chosen_test = random.sample(killing_tests_chosen, 1)[0]
            chosen_tests.append(chosen_test)
            to_run_tests += len(mutant_pool)
            mutant_pool = [mutant for mutant in mutant_pool
                           if chosen_test not in mutant.killingTests]
        ms_progression.append(mutation_score_with_tests(relevant_mutants, chosen_tests))
    return ms_progression[-1], mutants_picked, to_run_tests


def developer_simulation(data, path_to_mutants_data, output_path):
    for _index, row in data.iterrows():
        mutationMatrixPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutationMatrix.csv"
        mutantsInfoPath = path_to_mutants_data + "/" + row["project"] + "/" + _index + "/" + "mutants_info.csv"

        assert os.path.exists(mutationMatrixPath), "Does not exists: " + mutationMatrixPath
        assert os.path.exists(mutantsInfoPath), "Does not exists: " + mutantsInfoPath

        print(_index)
        all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants = map_mutants(
            mutants_info_path=mutantsInfoPath, mutation_matrix_path=mutationMatrixPath)

        minimal_mutants, subsumed_killed_mutants, mutants_killed, equivalent = calculate_minimal_mutants(all_granularity_level)
        subsumed_killed_mutants.update(equivalent)

        developer_simulation = output_path + "/developer_simulation.csv"
        with open(developer_simulation, "a+") as output_file:
            if os.stat(developer_simulation).st_size == 0:
                output_file.write("commit,target,mutant_pool,limit,iteration,mutants_picked,tests_picked,ms_progression\n")

            mutants_v2_killed = [mutant for mutant in all_granularity_level if len(mutant.killingTests) != 0]
            test_suite = set()
            [[test_suite.add(test) for test in mutant.killingTests] for mutant in mutants_v2_killed]

            for pool, name in [(all_granularity_level, "all"), (relevant_mutants, "relevant"), (on_change_mutants, "modification"), (minimal_relevant_mutants, "minimal_relevant"), (minimal_mutants, "minimal")]:
                for limit in range(2, 22, 2):
                    for i in range(1, 101):
                        ms_progress, muts_picks, chosen_tests = get_ms_from_simulation(pool, relevant_mutants,
                                                                                           test_suite,
                                                                                           limit)
                        write_ms_achieved(output_file, "Target_Relevant", _index, name, i, ms_progress, muts_picks, limit,
                                              chosen_tests)
                        ms_progress, muts_picks, chosen_tests = get_ms_from_simulation(pool, minimal_relevant_mutants,
                                                                                           test_suite,
                                                                                           limit)
                        write_ms_achieved(output_file, "Target_Minimal_Relevant", _index, name, i, ms_progress, muts_picks, limit,
                                          chosen_tests)

if __name__ == '__main__':
    arguments = parse_args().parse_args()

    data_frame = pd.read_csv(arguments.statistics_file, index_col="commit", thousands=",")

    data_frame = data_frame.drop(["3ee44535b4c6d925ae21cf8dcbe6e85a72158a68","dfd69e038cc7035031d1807c4ade870d2a7e2ece"])

    # OUTPUT FILES:
        # Mutation In Chronological order per project
        # Simulation on All Mutants
        # Split commits per categories and perform simulation

    venn_diagram_simulation(data=data_frame, path_to_mutants_data=arguments.path_to_mutants_data, output_path=arguments.output_dir)

    # developer_simulation(data=data_frame, path_to_mutants_data=arguments.path_to_mutants_data, output_path=arguments.output_dir)


