import argparse
import sys
import time
import os

sys.path.append("/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I")

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd


def get_arguments():
    parser = argparse.ArgumentParser(description="Train models for mutants prediction")
    parser.add_argument("-p", "--path_to_file",
                        action="store",
                        help="File where information about mutants is")
    parser.add_argument("-w", "--working_directory",
                        action="store",
                        default="/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I",
                        help="Script directory")
    parser.add_argument("-n", "--project_name", default="projects_all", action="store",
                        help="Load data from project specific file")
    parser.add_argument("-g", "--grid_search", action="store_true",
                        help="Choose whether to work on evolution of prediction")
    parser.add_argument("-r", "--random_search", action="store_true",
                        help="Choose whether to perform grid search random")
    parser.add_argument("-c", "--pickled", action="store_true",
                        help="Pickle a model")
    parser.add_argument("-x", "--path_to_pickled_model", action="store",
                        help="Set path to pickled model to be used")
    parser.add_argument("-s", "--save_figures", action="store_true",
                        help="Choose whether to save figures")
    parser.add_argument("-e", "--evolution", action="store_true",
                        help="Choose whether to work on evolution of prediction")

    arguments = parser.parse_args()
    print(arguments)
    return arguments


def initialise_usage_directories(working_directory, make_time_dirs=False):
    path_to_model = None
    logging.info("<<< Initialise output and input directories")
    logging.info("<<< Start")
    path_to_training_data = working_directory + "/" + os.path.join("data", "training")
    os.makedirs(path_to_training_data, exist_ok=True)
    print(path_to_training_data)
    if make_time_dirs:
        timestr = time.strftime("%Y%m%d-%H%M%S")

        path_to_model = path_to_training_data + "/" + os.path.join("models", timestr)
        os.makedirs(path_to_model, exist_ok=True)
        print(path_to_model)

    logging.info("<<< Done")
    return path_to_training_data, path_to_model


def load_data(path_to_file):
    logging.info(">>> Loading prepared data for: {path}".format(path=path_to_file))
    logging.info(">>> Start")

    projects = ["commons-collections", "commons-csv", "commons-io", "commons-lang", "commons-text"]
    prepared_string = "_prepared.csv"

    loaded_data = pd.read_csv(filepath_or_buffer=path_to_file,
                              index_col="MutantID", thousands=",",
                              parse_dates=["CommitDate"], low_memory=False)

    logging.info(">>> Done")
    return loaded_data

def data_preparation(data):
    logging.info(data.shape)
    print("\n\n")
    logging.info(data.columns)
    print("\n\n")
    logging.info(data.dtypes)
    print("\n\n")
    print(data.describe())
    print("\n\n")
    print(data['Relevant'].value_counts())
    print("\n\n")
    print(data['Minimal'].value_counts())
    print("\n\n")
    print(data.isna().sum())
    print()

if __name__ == '__main__':
    arguments = get_arguments()

    path_to_training_data, path_to_model = initialise_usage_directories(arguments.working_directory, make_time_dirs=False)

    # Load DATA
    data = load_data(path_to_file=arguments.path_to_file)

    data_preparation(data=data)

    # perform PCA to reduce number of features

    # Split data to X and Y

    # Perform Esamble training