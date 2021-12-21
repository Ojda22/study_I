import argparse
import csv
import os
from typing import Tuple, Type, List, Dict
import xml.etree.ElementTree as ET
import pandas as pd
import json
# from pydriller import RepositoryMining
# from pydriller.metrics.process.hunks_count import HunksCount


class MutantFile(object):

    def __init__(self, class_name: str, line_numbers: List, indexes: List, blocks: List, mutant_operators: List,
                 file_path: str):
        self.class_name = class_name
        self.lineNumbers = line_numbers
        self.indexes = indexes
        self.blocks = blocks
        self.mutant_operators = mutant_operators
        self.file_path = file_path
        self.sourceFile = ""
        self.mutatedMethod = []
        self.methodDescription = []
        self.assertions = []
        self.status = ""
        self.killingTests = frozenset()
        self.succidingTests = frozenset()
        self.prediction = 0.0
        self.mutant_ID = 0

    def to_string(self) -> str:
        # return "Name: {}\nLineNumbers: {}\nIndexes: {}\nBlocks: {}\nMutantOperators: {}\nFileName: {}\n".format(
        #     self.class_name,
        #     "-".join(self.lineNumbers),
        #     "-".join(self.indexes),
        #     "-".join(self.blocks),
        #     "-".join(
        #         self.mutant_operators),
        #     self.file_name())
        return "{} , {} , {} , {} , {} , {} , {}".format(
            self.class_name,
            "-".join(self.lineNumbers),
            "-".join(self.indexes),
            "-".join(self.blocks),
            "-".join(
                self.mutant_operators),
            "-".join(
                self.mutatedMethod),
            "-".join(
                self.methodDescription))

    def file_name(self) -> str:
        return self.file_path.split("/")[-1]

    def __ne__(self, other):
        return (not isinstance(other, type(self))
                or (self.class_name) != (other.class_name)
                or (self.lineNumbers) != (other.lineNumbers)
                or (self.indexes) != (other.indexes)
                or (self.blocks) != (other.blocks)
                or (self.mutant_operators) != (other.mutant_operators)
                or (self.mutatedMethod) != (other.mutatedMethod)
                or (self.methodDescription) != (other.methodDescription))

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.class_name) == (othr.class_name)
                and (self.lineNumbers) == (othr.lineNumbers)
                and (self.indexes) == (othr.indexes)
                and (self.blocks) == (othr.blocks)
                and (self.mutant_operators) == (othr.mutant_operators)
                and (self.mutatedMethod) == (othr.mutatedMethod)
                and (self.methodDescription) == (othr.methodDescription))

    def __hash__(self):
        return hash((self.class_name, ",".join(self.lineNumbers), ",".join(self.indexes), ",".join(self.blocks), ",".join(self.mutant_operators), ",".join(self.mutatedMethod), ",".join(self.methodDescription)))


class AssertionClazz(object):

    def __init__(self, assert_id: str, test_name: str, assert_value: str, text: str):
        self.assert_id = assert_id
        self.test_name = test_name
        self.assert_value = assert_value
        self.text = text
        self.exception_class = ""
        self.stacktrace = []
        self.test_exception_frame = ""

    def to_string(self) -> str:
        return "AssertID: {}\nTestName: {}\nAssertValue: {}\nText: {}\n".format(
            self.assert_id,
            self.test_name,
            self.assert_value,
            self.text)

    def __ne__(self, other):
        return (not isinstance(other, type(self))
                or (self.assert_id, self.test_name, self.assert_value, self.text) != (other.assert_id, other.test_name, other.assert_value, other.text))

    def __eq__(self, othr):
        return (isinstance(othr, type(self))
                and (self.assert_id, self.test_name, self.assert_value, self.text) == (othr.assert_id, othr.test_name, othr.assert_value, othr.text))

    def __hash__(self):
        return hash((self.assert_id, self.test_name, self.assert_value, self.text))


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Comparing mutants")
    parser.add_argument("-c", "--commitID", action="store", help="Put commit id")
    parser.add_argument("dir_fom", type=str, help="Directory of files with first order mutants")
    parser.add_argument("dir_som", type=str, help="Directory of files with second order mutants")
    parser.add_argument("changes_file", type=str, help="File with changes")
    parser.add_argument("-e", "--extract", action="store_true", help="Extract mutants")
    parser.add_argument("project_git_url", type=str, help="Target project git url")
    parser.add_argument("-o", "--output_file", action="store", help="Output file")
    parser.add_argument("-p", "--project_id", action="store", help="Project id")

    return parser


def validate_and_read_directories_with_mutant_files(dir_fom, dir_som) -> Tuple[List[str], List[str]]:
    if not os.path.isdir(arguments.dir_fom) or not os.path.isdir(dir_som):
        message = "Directories (or one of them) are not valid. Please check further: \ndir_fom: {}\ndir_som: {}".format(
            dir_fom, dir_som)
        raise Exception(message)

    print("Directory with first order mutants: {}".format(dir_fom))
    print("Directory with second order mutants: {}\n".format(dir_som))
    fom_files_in_list = os.listdir(dir_fom)
    som_files_in_list = os.listdir(dir_som)

    if len(fom_files_in_list) == 0 or len(som_files_in_list) == 0:
        message = "Directories (or one of them) are empty. Please check further: \ndir_fom: {}\ndir_som: {}".format(
            dir_fom, dir_som)
        raise Exception(message)

    print("Number of first order mutant files: {}".format(len(fom_files_in_list)))
    print("Number of second order mutant files: {}\n".format(len(som_files_in_list)))
    return fom_files_in_list, som_files_in_list


def contains(som_list, fom_mutant) -> List[MutantFile]:
    som_mutants_found = []
    for som_mutant in som_list:
        if fom_mutant.class_name == som_mutant.class_name and fom_mutant.lineNumbers[0] == som_mutant.lineNumbers[0] and \
                fom_mutant.indexes[0] == som_mutant.indexes[0] and fom_mutant.blocks[0] == som_mutant.blocks[0] \
                and fom_mutant.mutant_operators[0] == som_mutant.mutant_operators[0]:
            som_mutants_found.append(som_mutant)
    return som_mutants_found
    # message = "Second order mutant cannot find its first order mutant. This should not happen. Please check " \
    #           "further.\nSOM mutant: \n{}".format(som_mutant.to_string())
    # raise Exception(message)


# def match_fom_files_with_som_files(fom_files: List[MutantFile], som_files: List[MutantFile], data_changes) -> Tuple[
#     List[MutantFile], Dict[MutantFile, List[MutantFile]]]:
#     with_match = dict()
#     with_no_match = []
#     for fom_mutant_file in fom_files:
#         som_mutants = contains(som_files, fom_mutant_file)
#         if len(som_mutants) == 0:
#             if "$" in fom_mutant_file.class_name:
#                 clazz_name = fom_mutant_file.class_name.split("$")[0]
#             else:
#                 clazz_name = fom_mutant_file.class_name
#             for clazz, lines in data_changes.items():
#                 if clazz_name in clazz:
#                     if int(fom_mutant_file.lineNumbers[0]) in lines:
#                         with_no_match.append(fom_mutant_file)
#                         break
#                     break
#         else:
#             with_match[fom_mutant_file] = som_mutants
#     return with_no_match, with_match


def match_fom_files_with_som_files(fom_files: List[MutantFile], data_changes) -> List[MutantFile]:
    with_no_match = []
    for fom_mutant_file in fom_files:
        source_file = fom_mutant_file.sourceFile
        for changed_file, lines in data_changes.items():
            if source_file in changed_file:
                if int(fom_mutant_file.lineNumbers[0]) in lines:
                    with_no_match.append(fom_mutant_file)
                    break
                break
    return with_no_match


def objectifyy_mutant(class_name: str, line_numbers: List, indexes: List, blocks: List, mutant_operators: List, file_path: str):
    mutant = MutantFile(class_name=class_name, line_numbers=line_numbers, indexes=indexes, blocks=blocks,
                            mutant_operators=mutant_operators, file_path=file_path)

    assertions = []

    treeFOM = ET.parse(file_path)
    rootFOM = treeFOM.getroot()

    mutant.status = rootFOM.findall("mutation")[0].get("status")
    mutant.class_name = rootFOM[0].findall("mutatedClass")[0].text
    mutant.methodDescription = [description.text for description in rootFOM[0].findall("methodDescription")]
    mutant.mutatedMethod = [mutatedMethod.text for mutatedMethod in rootFOM[0].findall("mutatedMethod")]
    mutant.sourceFile = rootFOM[0].findall("sourceFile")[0].text

    # since the root is <mutations>, child is only one tag of <mutation> which we can statically take
    som_assertions = rootFOM[0].findall("assertion")

    for assertion in som_assertions:
        assertID = assertion.get("assertID")
        testName = assertion.get("testName")
        assertValue = assertion.get("assertValue")
        text = assertion.text
        assertion_object = AssertionClazz(assert_id=assertID, test_name=testName, assert_value=assertValue, text=text)
        assertions.append(assertion_object)

    mutant.assertions = assertions
    return mutant


def find_mutant(fom_mutants_list, som_mutant, level):
    for fom_mutant in fom_mutants_list:
        if som_mutant.lineNumbers[level] in fom_mutant.lineNumbers and som_mutant.indexes[level] in fom_mutant.indexes and som_mutant.blocks[level] in fom_mutant.blocks and fom_mutant.class_name == som_mutant.class_name and som_mutant.mutant_operators[level] in fom_mutant.mutant_operators:
            return fom_mutant
    return None


def parse_mutant_element(mutant_element, dir_som):
    # class_name = mutant_element.find("sourceFile").text
    class_name = mutant_element.find("mutatedClass").text
    line_numbers = [element.text for element in mutant_element.findall("lineNumber")]
    indexes = [element.text for element in mutant_element.findall("index")]
    blocks = [element.text for element in mutant_element.findall("block")]
    mutant_operators = [element.text for element in mutant_element.findall("mutator")]
    mutant = MutantFile(class_name=class_name, line_numbers=line_numbers, indexes=indexes, blocks=blocks,
                           mutant_operators=mutant_operators, file_path=dir_som)
    # mutant.sourceFile = mutant_element.find("mutatedClass").text
    mutant.sourceFile = mutant_element.find("sourceFile").text
    mutant.methodDescription = [element.text for element in mutant_element.findall("methodDescription")]
    mutant.mutatedMethod = [element.text for element in mutant_element.findall("mutatedMethod")]
    mutant.status = mutant_element.get("status")

    # since the root is <mutations>, child is only one tag of <mutation> which we can statically take
    som_assertions = mutant_element.findall("assertion")

    assertions = []
    killingTests = set()
    allTests = set()

    for assertion in som_assertions:
        assertID = assertion.get("assertID")
        # testName = assertion.get("testName")
        testCasePath = assertion.get("testName")
        if "(" in testCasePath:
            testCasePath = testCasePath.split("(")[0]
        assertValue = assertion.get("assertValue")
        text = assertion.text
        assertion_object = AssertionClazz(assert_id=assertID, test_name=testCasePath, assert_value=assertValue,
                                          text=text)

        if assertID == "[EXCEPTION]":
        # split by stacktrace if exists compare stack and compare exception
            if "[STACKTRACE]" in text:
                exception_stacktrace = text.split("[STACKTRACE]")
                exception_class = exception_stacktrace[0].strip()
                assertion_object.exception_class = exception_class

                # Split on the last bracket (intended space does not work) -- in future write stacktrace in a more suitable manner
                # fom_stack_trace = map(( lambda x: x + ")"), exception_stacktrace[1].split(")"))
                stack_trace_full = exception_stacktrace[1].strip().split(")")
                # take frames from the test package - by artifact id and groupid
                test_package_group = testCasePath.split(".")[:4]
                test_package_name = ".".join(test_package_group)
                test_stack_trace = [frame.strip() for frame in stack_trace_full if test_package_name in frame]

                assertion_object.stacktrace = test_stack_trace
                for test_exception_frame in test_stack_trace:
                    frame = test_exception_frame.split("(")
                    testName = frame[0].split(".")[-1]
                    if "Test" in testName or "test" in testName:
                        assertion_object.test_exception_frame = testName
                        break

            if "_ESTest" in testCasePath:
                test = testCasePath + "." + assertion_object.test_exception_frame
            else:
                test = testCasePath
        else:
            if "_ESTest" in testCasePath:
                testInAssertID = assertID.split(":")[1]
                test = testCasePath + "." + testInAssertID
            else:
                test = testCasePath

        if assertValue == "false":
            killingTests.add(test)
        else:
            allTests.add(test)
        assertions.append(assertion_object)

    mutant.assertions = assertions
    mutant.killingTests = frozenset([killingTest for killingTest in killingTests if killingTest[-1] != "."])
    succidingTests = frozenset([succeding_test for succeding_test in allTests if
                                succeding_test not in killingTests and succeding_test[-1] != "."])
    mutant.succidingTests = succidingTests

    return mutant


# def map_foms_to_som(dir_fom: str, dir_som: str, fom_files_list: List[str], som_file_list: List[str]) -> Dict[MutantFile, List[MutantFile]]:
def map_foms_to_som(dir_fom: str, dir_som: str):
    som_mutants_dict = dict()
    missing_mutant_files = set()

    mutants_y = set()

    fom_mutants = []

    treeFOM = ET.parse(dir_fom)
    rootFOM = treeFOM.getroot()

    counter_ID = 1
    for mutant in rootFOM.findall("mutation"):
        mutantObject = parse_mutant_element(mutant, dir_fom)
        mutantObject.mutant_ID = counter_ID
        fom_mutants.append(mutantObject)
        counter_ID += 1

    treeSOM = ET.parse(dir_som)
    rootSOM = treeSOM.getroot()

    for som_mutant in rootSOM:
        xy_mutant = parse_mutant_element(som_mutant, dir_som)

        x_mutant = find_mutant(fom_mutants, xy_mutant, level=0)
        y_mutant = find_mutant(fom_mutants, xy_mutant, level=1)

        if x_mutant is None:
            message = "Mutant is not executed because there is no tests in order. Please check further\nMutant XY: {}\n".format(xy_mutant.to_string())
            missing_mutant_files.add(x_mutant)
            continue
        if y_mutant is None:
            message = "Mutant is not executed because there is no tests in order. Please check further\nMutant XY: {}\n".format(
                xy_mutant.to_string())
            missing_mutant_files.add(y_mutant)
            continue

        mutants_y.add(y_mutant)
        som_mutants_dict[xy_mutant] = [x_mutant, y_mutant]

    return som_mutants_dict, missing_mutant_files, fom_mutants, mutants_y



def parse_files_name(dir_som: str, som_files_list: List[str]) -> List[MutantFile]:
    mutants_files = list()

    for file_name_as_mutant_id in som_files_list:
        file_path = os.path.join(dir_som, file_name_as_mutant_id)
        if not os.path.isfile(file_path):
            message = "File is not regular. Please check further\nFile path: {}".format(file_path)
            raise Exception(message)

        mutant_attributes = file_name_as_mutant_id.split("-")

        if len(mutant_attributes) != 5:
            message = "File format is not regular. Please check further\nFile name: {}".format(file_name_as_mutant_id)
            raise Exception(message)

        file = MutantFile(mutant_attributes[0], mutant_attributes[1].split(","), mutant_attributes[2].split(","),
                          mutant_attributes[3].split(","),
                          mutant_attributes[4].split(","), file_path)

        mutants_files.append(file)

    return mutants_files


def assertions_for_specific_test(assertions, test_name: str):
    assertions_matching_test_name = []
    for assertion in assertions:
        if assertion.get("assertValue") == test_name:
            assertions_matching_test_name.append(assertion)
    return assertions_matching_test_name


def mapping_assertions_to_tests(fom_assertions_list, som_assertions_list):
    fom_tests_list = []
    fom_assertions_to_test_dictionary = dict()
    som_assertions_to_test_dictionary = dict()
    # mapping a test with its assertions
    for assertionFOM in fom_assertions_list:
        if assertionFOM.get("testName") not in fom_tests_list:
            fom_tests_list.append(assertionFOM.get("testName"))
            # find assertions with this test name, and add them into a list
            # both for fom_mutant and som_mutant
            fom_assertions_for_test_list = assertions_for_specific_test(fom_assertions_list,
                                                                        assertionFOM.get("testName"))
            som_assertions_for_test_list = assertions_for_specific_test(som_assertions_list,
                                                                        assertionFOM.get("testName"))

            fom_assertions_to_test_dictionary[fom_tests_list[-1]] = fom_assertions_for_test_list
            som_assertions_to_test_dictionary[fom_tests_list[-1]] = som_assertions_for_test_list

            if len(som_assertions_for_test_list) == 0:
                message = "\nThere is no assertions in SOM for test: {}".format(fom_tests_list[-1])
                raise Exception(message)
    return fom_tests_list, fom_assertions_to_test_dictionary, som_assertions_to_test_dictionary


def find_assertion_by(assertions, element_name, element_value):
    matching_elements = []
    for _ in assertions:
        if _.get(element_name) == element_value:
            matching_elements.append(_)
    return matching_elements


def find_assertion_by_name_and_exception(assertions, element_name, element_value, exception):
    matching_elements = []
    for _ in assertions:
        if _.get(element_name) == element_value and _.get("assertID") == exception:
            matching_elements.append(_)
    return matching_elements


def find_assertion_by_id_test(assertions, assertID, testName, test_exception_frame):
# def find_assertion_by_id_test(assertions, assertID, testName):
    matching_elements = []
    for assertion in assertions:
        if assertion.assert_id == assertID and assertion.test_name == testName and assertion.test_exception_frame == test_exception_frame:
        # if assertion.assert_id == assertID and assertion.test_name == testName:
            matching_elements.append(assertion)

    if len(matching_elements) == 0:
        return None

    if len(matching_elements) > 1:
        message = "\nAssertion is either not found or there is many assertions with the same id. AssertID: {}, TestName: {}".format(assertID, testName)
        raise Exception(message)

    return matching_elements[0]


# def metrices_per_file_changed(commit_id, project_git_url):
#     changed_files = dict()
#     num_of_files = 0
#     total_complexity = 0
#     total_added_lines = 0
#     total_hunks = 0
#     total_loc = 0
#     for commit in RepositoryMining(path_to_repo=project_git_url, single=commit_id).traverse_commits():
#         num_of_files = len(commit.modifications)
#         for m in commit.modifications:
#             changed_files[m.filename] = {"file_complexity":m.complexity, "loc":m.nloc, "added_lines":m.added, "tokens":m.token_count}
#             total_complexity += m.complexity
#             total_added_lines += m.added
#             total_loc += m.nloc
#
#     hunks = HunksCount(path_to_repo=project_git_url, from_commit=commit_id, to_commit=commit_id)
#     hunks_dict = hunks.count()
#
#     for file, hunks_number in hunks_dict.items():
#         total_hunks += hunks_number
#         file = file.split("/")[-1]
#         if(file in changed_files.keys()):
#             changed_files[file].update({"hunks":hunks_number})
#
#     changed_files["change_overall"] = {"num_of_files":num_of_files, "total_added_lines":total_added_lines, "total_complexity":total_complexity, "total_hunks":total_hunks, "total_loc":total_loc}
#     return changed_files


# def createMutantsDataframe(mutants_list, mutants_on_line, minimal_mutants_relevant, df_columns, commitID, project_git_url):
#     rows = []
#     changed_files_dict = metrices_per_file_changed(commitID, project_git_url)
#     for mutant in mutants_list:
#         sourceFile = mutant.sourceFile
#         mutatedClass = mutant.class_name
#         mutatedMethod = mutant.mutatedMethod[0]
#         lineNumber = mutant.lineNumbers[0]
#         index = mutant.indexes[0]
#         block = mutant.blocks[0]
#         operator = mutant.mutant_operators[0]
#         methodDescription = mutant.methodDescription[0]
#
#         minimal = 0
#         if minimal_mutants_relevant is not None:
#             if mutant in minimal_mutants_relevant:
#                 minimal = 1
#
#         # changed_file_dict = changed_files_dict[mutatedClass]
#         changed_file_dict = changed_files_dict[sourceFile]
#         changed_files_overall = changed_files_dict["change_overall"]
#
#         on_line = 0
#
#         rows.append({"sourceFile": sourceFile, "mutatedClass": mutatedClass, "mutatedMethod": mutatedMethod,
#                      "lineNumber": lineNumber, "index": index, "onLine": on_line, "minimal" : minimal,
#                      "block": block, "mutator": operator, "methodDescription": methodDescription,
#                      "file_complexity": str(changed_file_dict["file_complexity"]), "nloc":str(changed_file_dict["loc"]), "added_lines":str(changed_file_dict["added_lines"]), "tokens":str(changed_file_dict["tokens"]), "hunks":str(changed_file_dict["hunks"]),
#                      "num_of_files":str(changed_files_overall["num_of_files"]), "total_added_lines":str(changed_files_overall["total_added_lines"]), "total_complexity":str(changed_files_overall["total_complexity"]), "total_hunks":str(changed_files_overall["total_hunks"]), "total_loc":str(changed_files_overall["total_loc"])})
#
#     if mutants_on_line is not None:
#         for mutant in mutants_on_line:
#             sourceFile = mutant.sourceFile
#             mutatedClass = mutant.class_name
#             mutatedMethod = mutant.mutatedMethod[0]
#             lineNumber = mutant.lineNumbers[0]
#             index = mutant.indexes[0]
#             block = mutant.blocks[0]
#             operator = mutant.mutant_operators[0]
#             methodDescription = mutant.methodDescription[0]
#
#             # changed_file_dict = changed_files_dict[mutatedClass]
#             changed_file_dict = changed_files_dict[sourceFile]
#             changed_files_overall = changed_files_dict["change_overall"]
#
#             on_line = 1
#
#             rows.append({"sourceFile": sourceFile, "mutatedClass": mutatedClass, "mutatedMethod": mutatedMethod,
#                          "lineNumber": lineNumber, "index": index, "onLine": on_line, "minimal" : 0,
#                          "block": block, "mutator": operator, "methodDescription": methodDescription,
#                          "file_complexity": str(changed_file_dict["file_complexity"]),
#                          "nloc": str(changed_file_dict["loc"]), "added_lines": str(changed_file_dict["added_lines"]),
#                          "tokens": str(changed_file_dict["tokens"]), "hunks": str(changed_file_dict["hunks"]),
#                          "num_of_files": str(changed_files_overall["num_of_files"]),
#                          "total_added_lines": str(changed_files_overall["total_added_lines"]),
#                          "total_complexity": str(changed_files_overall["total_complexity"]),
#                          "total_hunks": str(changed_files_overall["total_hunks"]),
#                          "total_loc": str(changed_files_overall["total_loc"])})
#
#     dataframe = pd.DataFrame(rows, columns=df_columns)
#     return dataframe

def find_intersection_of_assertions(X_assertions, Y_assertions):
    x_assertions = []
    y_assertions = []
    for assertion_X in X_assertions:
        for assertion_Y in Y_assertions:
            if assertion_X.assert_id == assertion_Y.assert_id and assertion_X.test_name == assertion_Y.test_name and assertion_X.test_exception_frame == assertion_Y.test_exception_frame:
            # if assertion_X.assert_id == assertion_Y.assert_id and assertion_X.test_name == assertion_Y.test_name:
                x_assertions.append(assertion_X)
                y_assertions.append(assertion_Y)
                break
    return zip(x_assertions, y_assertions)


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


def get_minimal_mutants(mutants):
    mutants_killed = [mutant for mutant in mutants if len(mutant.killingTests) != 0]
    equivalent = [mutant for mutant in mutants if len(mutant.killingTests) == 0 and len(mutant.succidingTests) == 0]
    equivalent_not_killed = [mutant for mutant in mutants if len(mutant.killingTests) == 0]
    killing_tests = set()
    for mutant in mutants_killed:
        if mutant.killingTests not in killing_tests:
            # mutants_unique.add(mutant)
            killing_tests.add(mutant.killingTests)

    minimal_mutants = set(mutants_killed)
    for mutant in mutants_killed:
        minimal_mutants = minimal_mutants - set([m for m in minimal_mutants if mutant.killingTests.issubset(
            m.killingTests) and mutant.killingTests != m.killingTests])

    return [minimal_mutants, set(mutants_killed) - minimal_mutants, mutants_killed, equivalent_not_killed]



if __name__ == '__main__':
    arguments = parse_args().parse_args();

    with open(arguments.changes_file) as json_file:
        data = json.load(json_file)

    # dictionary of second order mutant and list of mutants it is consisted of i.e. mutant around the change, mutant on change
    foms_to_som_map, missing_mutant_files, fom_mutants, mutants_on_line = map_foms_to_som(arguments.dir_fom, arguments.dir_som)

    # mutants_on_line = match_fom_files_with_som_files(fom_mutants, data)

    timed_outed_or_unknown_pit_behaviour_mutants = set()
    mutants_x_and_xy_are_different = set()
    mutant_satisfies_dif_output_but_not_killed = set()
    mutant_killed_with_different_exception = set()
    relevant_mutants = set()
    not_relevant_mutants = set()
    all_first_order = set()

    relevant_on_line = set()

    for XY, X_and_Y in foms_to_som_map.items():
        X = X_and_Y[0]
        Y = X_and_Y[1]

        # We need to create intersection set of assertions, we need to compare just those tests that are covering both X and Y
        zip_x_y_assertions = find_intersection_of_assertions(X.assertions, Y.assertions)

        relevant = False
        for x_assertion, y_assertion in zip_x_y_assertions:

            som_assertion = find_assertion_by_id_test(XY.assertions, x_assertion.assert_id, x_assertion.test_name, x_assertion.test_exception_frame)
            # som_assertion = find_assertion_by_id_test(XY.assertions, x_assertion.assert_id, x_assertion.test_name)
            if som_assertion is None:
                timed_outed_or_unknown_pit_behaviour_mutants.add(X)
                break

            # >>> SCENARIO <<<
            # if a mutant is killed with a test that has throwed an exception, we exclude this scenarios for further observation
            if x_assertion.assert_id == ["EXCEPTION"] and y_assertion.assert_id == ["EXCEPTION"] and som_assertion.assert_id == ["EXCEPTION"]:
                if x_assertion.exception_class != som_assertion.exception_class and y_assertion.exception_class != som_assertion.exception_class:
                    mutant_killed_with_different_exception.add(X)
                    break
                else:
                    if x_assertion.stacktrace != som_assertion.stacktrace and y_assertion.stacktrace != som_assertion.stacktrace:
                        mutant_killed_with_different_exception.add(X)
                        break

            if x_assertion.text != som_assertion.text:
                mutants_x_and_xy_are_different.add(X)
                if y_assertion.text != som_assertion.text:
                    if X.status == "KILLED":
                        relevant = True
                        break
                    # mutant_satisfies_dif_output_but_not_killed.add(X)
            #         else:
            #             relevant = False
            #             break
            #     else:
            #         relevant = False
            #         break
            # else:
            #     relevant = False
            #     break
            mutant_satisfies_dif_output_but_not_killed.add(X)

        if relevant:
            relevant_mutants.add(X)
            relevant_on_line.add(Y)

        all_first_order.add(X)

    not_relevant_mutants = [mutant for mutant in all_first_order if mutant not in relevant_mutants and mutant not in mutants_on_line]

    intersection = relevant_on_line.intersection(mutants_on_line)
    print(intersection)

    # print("\nNumber of first order mutants: {}".format(len(fom_mutants)))
    # print("Number of mutants on change: {}".format(len(mutants_on_line)))
    # print("Number of second order mutant: {}".format(len(foms_to_som_map.keys())))
    # print("Number of first order mutants not generated because there is no tests in order for them: {}".format(len(missing_mutant_files)))
    # print("Number of first ordered class granularity observed: {}".format(len(all_first_order)))
    # print("Number of relevant mutants: {}".format(len(relevant_mutants)))
    # print("Number of not relevant mutants: {}".format(len(not_relevant_mutants)))
    # print("Number of mutants not killed but with relevant assertion difference: {}".format(len(mutant_satisfies_dif_output_but_not_killed)))
    # print("Number of mutants with assertion difference x != xy: {}".format(len(mutants_x_and_xy_are_different)))
    # print("Number of mutants timed outed or unknow pit behaviour (assertion does exist in SOM): {}".format(len(timed_outed_or_unknown_pit_behaviour_mutants)))
    # print("Number of mutants which are all killed with different exception (X, Y and XY): {}".format(len(mutant_killed_with_different_exception)))

    # print("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}".format(arguments.commitID,
    #                                                              len(fom_mutants),
    #                                                              len(mutants_on_line),
    #                                                              len(foms_to_som_map.keys()),
    #                                                              len(all_first_order),
    #                                                              len(relevant_mutants),
    #                                                              len(not_relevant_mutants),
    #                                                              len(mutant_satisfies_dif_output_but_not_killed),
    #                                                              len(mutants_x_and_xy_are_different),
    #                                                              len(timed_outed_or_unknown_pit_behaviour_mutants),
    #                                                              len(mutant_killed_with_different_exception)))

    all_first_order.update(mutants_on_line)
    minimal_mutants, subsumed_mutants, mutants_killed, equivalent = get_minimal_mutants(all_first_order)
    minimal_mutants_relevant, subsumed_mutants_relevant, mutants_killed_relevant, equivalent_relevant = get_minimal_mutants(
        relevant_mutants)

    killingTestsUnion = set()
    for m in fom_mutants:
        killingTestsUnion.update(set([test for test in m.killingTests]))

    mutants_info = []
    mutation_matrix = []
    mutation_matrix_columns = list(killingTestsUnion)
    mutation_matrix_columns.insert(0, "MutantID")
    for first_order_mutant in fom_mutants:
        is_relevant = 0
        is_not_relevant = 0
        is_on_changed_line = 0
        is_minimal_relevant = 0

        if first_order_mutant in mutants_on_line:
            is_on_changed_line = 1
        elif first_order_mutant in not_relevant_mutants:
            is_not_relevant = 1
        elif first_order_mutant in relevant_mutants:
            is_relevant = 1
            if first_order_mutant in minimal_mutants_relevant:
                is_minimal_relevant = 1


        mutants_info.append({"MutantID" : first_order_mutant.mutant_ID,
                             "Relevant" : is_relevant,
                             "Not_relevant" : is_not_relevant,
                             "Minimal_relevant" : is_minimal_relevant,
                             "On_Change": is_on_changed_line,
                             "sourceFile": first_order_mutant.sourceFile, "mutatedClass": first_order_mutant.class_name, "mutatedMethod": first_order_mutant.mutatedMethod[0],
                             "lineNumber": first_order_mutant.lineNumbers[0], "index": first_order_mutant.indexes[0],
                             "block": first_order_mutant.blocks[0], "mutator": first_order_mutant.mutant_operators[0], "methodDescription": first_order_mutant.methodDescription[0]
                             })

        killingMatrixForMutant = { "MutantID": first_order_mutant.mutant_ID}
        for test in killingTestsUnion:
            if test in first_order_mutant.killingTests:
                killingMatrixForMutant.update({test: 1})
            else:
                killingMatrixForMutant.update({test: 0})
        mutation_matrix.append(killingMatrixForMutant)

    mutation_matrix_file = "./mutationMatrix_hard.csv"
    print("Outputing mutation matrix: {}".format(mutation_matrix_file))
    try:
        with open(mutation_matrix_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=mutation_matrix_columns)
            writer.writeheader()
            for data in mutation_matrix:
                writer.writerow(data)
    except IOError:
        print("Finished outputing mutation matrix")

    mutants_info_columns = ["MutantID", "Relevant", "Not_relevant", "Minimal_relevant", "On_Change", "sourceFile", "mutatedClass", "mutatedMethod", "lineNumber",
                  "index",
                  "block", "mutator", "methodDescription"]

    mutants_info_file = "./mutants_info_hard.csv"
    print("Outputing mutants info: {}".format(mutants_info_file))
    try:
        with open(mutants_info_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=mutants_info_columns)
            writer.writeheader()
            for data in mutants_info:
                writer.writerow(data)
    except IOError:
        print("Finished outputing mutation matrix")
    #
    mutants_v2_killed = [mutant for mutant in all_first_order if len(mutant.killingTests) != 0]
    test_suite = set()
    [[test_suite.add(test) for test in mutant.killingTests] for mutant in mutants_v2_killed]

    test_suite_relevant = set()
    [[test_suite_relevant.add(test) for test in mutant.killingTests] for mutant in relevant_mutants]

    ms = mutation_score_with_tests(all_first_order, test_suite)

    with open(arguments.output_file + "/statistics__hard_" + arguments.project_id + ".csv", "a+") as output_file:
        if os.stat(arguments.output_file + "/statistics__hard_" + arguments.project_id + ".csv").st_size == 0:
            output_file.write(
                "commit,fom_mutants,mutants_on_change,som_mutants,mutants_gran,relevant_mutants,not_relevant_mutants,minimal_mutants,equivalent,minimal_relevant_mutants,MS,total_tests,relevant_tests\n")

        output_file.write("{commit},{fom_mutants},{mutants_on_change},{som_mutants},{mutants_gran},{relevant_mutants},{not_relevant_mutants},{minimal_mutants},{equivalent},{minimal_relevant_mutants},{MS},{total_tests},{relevant_tests}".format(
                                         commit=arguments.commitID,
                                         fom_mutants=len(fom_mutants),
                                         mutants_on_change=len(mutants_on_line),
                                         som_mutants=len(foms_to_som_map.keys()),
                                         mutants_gran=len(all_first_order),
                                         relevant_mutants=len(relevant_mutants),
                                         not_relevant_mutants=len(not_relevant_mutants),
                                         minimal_mutants=len(minimal_mutants),
                                         equivalent=len(equivalent),
                                         minimal_relevant_mutants=len(minimal_mutants_relevant),
                                         MS=ms,
                                         total_tests=len(test_suite),
                                         relevant_tests=len(test_suite_relevant)
                                         ))
        output_file.write("\n")

    if arguments.extract:
        df_columns = ["sourceFile", "mutatedClass", "mutatedMethod", "lineNumber", "index", "onLine", "minimal", "block", "mutator", "methodDescription",
                    "file_complexity", "nloc", "added_lines", "tokens",
                    "num_of_files", "total_added_lines", "total_complexity", "total_hunks", "total_loc"]


        # relevantMutantsDF = createMutantsDataframe(relevant_mutants, mutants_on_line, minimal_mutants_relevant, df_columns, arguments.commitID, arguments.project_git_url)
        # not_relevantMutantsDF = createMutantsDataframe(not_relevant_mutants, None, None, df_columns, arguments.commitID, arguments.project_git_url)

        # with open("./minimal_relevant_mutants_hard.json", "w") as fileR:
        #     relevantMutantsDF.to_json(fileR, orient="records")
        #
        # with open("./minimal_not_relevant_mutants_hard.json", "w") as fileR:
        #     not_relevantMutantsDF.to_json(fileR, orient="records")

