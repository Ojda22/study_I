import argparse
import pandas as pd
import numpy as np

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comparing mutants")
    parser.add_argument("-c", "--commitID", action="store", help="Put commit id")
    parser.add_argument("-i", "--mutants_info", action="store", help="Put mutantsInfo.csv file")
    parser.add_argument("-m", "--mutation_matrix", action="store", help="Put mutationMatrix.csv file")

    return parser


class Mutant(object):

    def __init__(self):
        self.class_name = ""
        self.lineNumber = ""
        self.index = ""
        self.block = ""
        self.mutant_operator = ""
        self.sourceFile = ""
        self.mutatedMethod = ""
        self.methodDescription = ""
        self.killingTests = frozenset()
        self.succidingTests = frozenset()
        self.prediction = 0.0
        self.mutant_ID = 0
        self.is_relevant = 0
        self.is_on_change = 0
        self.is_minimal = 0
        self.is_not_relevant = 0

    def hard_to_kill_score(self) -> float:
        if len(self.killingTests) == 0:
            return 1
        return len(self.killingTests) / (len(self.killingTests) + len(self.succidingTests))

    def to_string(self) -> str:
        return "{} , {} , {} , {} , {} , {} , {}".format(
            self.class_name,
            self.lineNumber,
            self.index,
            self.block,
            self.mutant_operator,
            self.mutatedMethod,
            self.methodDescription)

    def __ne__(self, other):
        return (not isinstance(other, type(self))
                or (self.class_name) != (other.class_name)
                or (self.lineNumber) != (other.lineNumber)
                or (self.index) != (other.index)
                or (self.block) != (other.block)
                or (self.mutant_operator) != (other.mutant_operator)
                or (self.mutatedMethod) != (other.mutatedMethod)
                or (self.methodDescription) != (other.methodDescription))

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.class_name) == (othr.class_name)
                and (self.lineNumber) == (othr.lineNumber)
                and (self.index) == (othr.index)
                and (self.block) == (othr.block)
                and (self.mutant_operator) == (othr.mutant_operator)
                and (self.mutatedMethod) == (othr.mutatedMethod)
                and (self.methodDescription) == (othr.methodDescription))

    def __hash__(self):
        return hash((self.class_name, self.lineNumber, self.index, self.block, self.mutant_operator, self.mutatedMethod, self.methodDescription))


def calculate_minimal_mutants(mutants):
    mutants_killed = [mutant for mutant in mutants if len(mutant.killingTests) != 0]
    # equivalent = [mutant for mutant in mutants if len(mutant.killingTests) == 0 and len(mutant.succidingTests) == 0]
    equivalent_not_killed = [mutant for mutant in mutants if len(mutant.killingTests) == 0]
    killing_tests = set()
    for mutant in mutants_killed:
        if mutant.killingTests not in killing_tests:
            killing_tests.add(mutant.killingTests)

    minimal_mutants = set(mutants_killed)
    for mutant in mutants_killed:
        minimal_mutants = minimal_mutants - set([m for m in minimal_mutants if mutant.killingTests.issubset(
            m.killingTests) and mutant.killingTests != m.killingTests])

    return [minimal_mutants, set(mutants_killed) - minimal_mutants, mutants_killed, equivalent_not_killed]


def map_mutants(mutants_info_path, mutation_matrix_path):
    mutants_info_df = pd.read_csv(filepath_or_buffer=mutants_info_path, index_col="MutantID", delimiter=",")
    mutation_matrix_df = pd.read_csv(filepath_or_buffer=mutation_matrix_path, index_col="MutantID", delimiter=",")

    all_fom_mutants = []
    for _index, row in mutation_matrix_df.iterrows():
        mutant = Mutant()
        mutant.mutant_ID = _index
        mutant.killingTests = frozenset([r[0] for r in row.iteritems() if r[1] == 1])
        mutant.succidingTests = frozenset([r[0] for r in row.iteritems() if r[1] == 0])
        mutant_info = mutants_info_df.loc[[_index]]
        mutant.is_relevant = mutant_info["Relevant"].iloc[0]
        mutant.is_not_relevant = mutant_info["Not_relevant"].iloc[0]
        mutant.is_minimal = mutant_info["Minimal_relevant"].iloc[0]
        mutant.is_on_change = mutant_info["On_Change"].iloc[0]
        mutant.sourceFile = mutant_info["sourceFile"].iloc[0]
        mutant.class_name = mutant_info["mutatedClass"].iloc[0]
        mutant.mutatedMethod = mutant_info["mutatedMethod"].iloc[0]
        mutant.lineNumber = mutant_info["lineNumber"].iloc[0]
        mutant.index = mutant_info["index"].iloc[0]
        mutant.block = mutant_info["block"].iloc[0]
        mutant.mutant_operator = mutant_info["mutator"].iloc[0]
        mutant.methodDescription = mutant_info["methodDescription"].iloc[0]
        all_fom_mutants.append(mutant)

    relevant_mutants = [m for m in all_fom_mutants if m.is_relevant == 1 ]
    not_relevant_mutants = [m for m in all_fom_mutants if m.is_not_relevant == 1 ]
    on_change_mutants = [m for m in all_fom_mutants if m.is_on_change == 1 ]
    minimal_relevant_mutants = [m for m in all_fom_mutants if m.is_minimal == 1 ]

    all_granularity_level = relevant_mutants + not_relevant_mutants + on_change_mutants

    return all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants


def return_tests(mutants_info_path, mutation_matrix_path):
    mutants_info_df = pd.read_csv(filepath_or_buffer=mutants_info_path, index_col="MutantID", delimiter=",")
    mutation_matrix_df = pd.read_csv(filepath_or_buffer=mutation_matrix_path, index_col="MutantID", delimiter=",")

    developer_tests = set()
    generated_tests = set()
    for _index, row in mutation_matrix_df.iterrows():
        for r in row.iteritems():
            if "ESTest" in r[0]:
                generated_tests.add(r[0])
            else:
                developer_tests.add(r[0])

    return developer_tests, generated_tests


def calculate_relevant_ms(mutants_info_path, mutation_matrix_path):
    mutants_info_df = pd.read_csv(filepath_or_buffer=mutants_info_path, index_col="MutantID", delimiter=",").to_dict(
        "index")
    mutation_matrix_df = pd.read_csv(filepath_or_buffer=mutation_matrix_path, index_col="MutantID", delimiter=",")

    killed_dev = 0
    survived_dev = 0

    for _index, row in mutation_matrix_df.iterrows():
        killed_dev_label = False

        mutant = mutants_info_df.get(_index)
        if mutant["Minimal_relevant"] != 1:
            continue

        for r in row.iteritems():
            if "ESTest" not in r[0]:
                if r[1] == 1:
                    killed_dev_label = True

        if killed_dev_label:
            killed_dev += 1
        else:
            survived_dev += 1

    return round(killed_dev / (survived_dev + killed_dev) * 100, 2)


def calculate_ms(mutants_info_path, mutation_matrix_path):
    mutants_info_df = pd.read_csv(filepath_or_buffer=mutants_info_path, index_col="MutantID", delimiter=",").to_dict("index")
    mutation_matrix_df = pd.read_csv(filepath_or_buffer=mutation_matrix_path, index_col="MutantID", delimiter=",")

    killed_dev = 0
    survived_dev = 0
    killed_evo = 0
    survived_evo = 0

    killed = 0
    survived = 0

    for _index, row in mutation_matrix_df.iterrows():
        killed_dev_label = False
        killed_evo_label = False

        for r in row.iteritems():
            if "ESTest" not in r[0]:
                if r[1] == 1:
                    killed_dev_label = True
            else:
                if r[1] == 1:
                    killed_evo_label = True

        if killed_dev_label:
            killed_dev += 1
        else:
            survived_dev += 1

        if killed_evo_label:
            killed_evo += 1
        else:
            survived_evo += 1

        if killed_evo_label or killed_dev_label:
            killed += 1
        else:
            survived += 1

    return round(killed_dev / (survived_dev + killed_dev) * 100, 2), round(killed_evo / (survived_evo + killed_evo) * 100, 2), round(killed / (survived + killed) * 100, 2)

if __name__ == '__main__':
    arguments = parse_args().parse_args()

    mutants_info_df = pd.read_csv(filepath_or_buffer=arguments.mutants_info, index_col="MutantID", delimiter=",")
    mutation_matrix_df = pd.read_csv(filepath_or_buffer=arguments.mutation_matrix, index_col="MutantID", delimiter=",")

    all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants = map_mutants(mutants_info_path=arguments.mutants_info, mutation_matrix_path=arguments.mutation_matrix)

    print()