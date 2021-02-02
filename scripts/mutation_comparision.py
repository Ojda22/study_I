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
        self.prediction = 0.0
        self.mutant_ID = 0
        self.is_relevant = 0
        self.is_on_change = 0
        self.is_minimal = 0
        self.is_not_relevant = 0

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



def map_mutants(mutants_info_dataframe, mutation_matrix_dataframe):
    all_fom_mutants = []
    for _index, row in mutation_matrix_df.iterrows():
        mutant = Mutant()
        mutant.mutant_ID = _index
        mutant.killingTests = set([r[0] for r in row.iteritems() if r[1] == 1])
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

    print("All fom mutants: {number}".format(number=len(all_fom_mutants)))
    print("All granularity: {number}".format(number=len(all_granularity_level)))
    print("Relevant mutants: {number}".format(number=len(relevant_mutants)))
    print("Not relevant mutants: {number}".format(number=len(not_relevant_mutants)))
    print("On change mutants: {number}".format(number=len(on_change_mutants)))
    print("Minimal relevant mutants: {number}".format(number=len(minimal_relevant_mutants)))

    return all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants

if __name__ == '__main__':
    arguments = parse_args().parse_args()

    mutants_info_df = pd.read_csv(filepath_or_buffer=arguments.mutants_info, index_col="MutantID", delimiter=",")
    mutation_matrix_df = pd.read_csv(filepath_or_buffer=arguments.mutation_matrix, index_col="MutantID", delimiter=",")

    all_fom_mutants, all_granularity_level, relevant_mutants, not_relevant_mutants, on_change_mutants, minimal_relevant_mutants = map_mutants(mutants_info_dataframe=mutants_info_df, mutation_matrix_dataframe=mutation_matrix_df)

    print()