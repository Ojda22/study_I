import argparse
import sys
import time
import os

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score, \
    matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# sys.path.append("/Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/study_I")
sys.path.append("/home/mojdanic/study_I")

import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import numpy as np
import matplotlib.pyplot as plt

import xgboost as xgb

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

def data_preparation(data, output_path):
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
    print(data["Project"].unique())
    print()

    projects = ['commons-lang', 'commons-io', 'commons-csv', 'commons-text', 'commons-collections']
    # projects = ['commons-io', 'commons-csv', 'commons-text', 'commons-collections']

    for project in projects:

        data_project = data[data["Project"] == project]

        print("Project: ", project)
        print(data_project.shape)

        identification_features = ["CommitDate", "Project", "Mutator",
                                   "SourceFile", "MutatedClass", "MutatedMethod", "LineNumber", "Index", "Block",
                                   "MethodDescription"]
        identification_data = data_project[identification_features]
        data_project = data_project.drop(identification_features, axis=1)

        Y = data_project[["Minimal", "Relevant"]]
        X = data_project.drop(["Minimal", "Relevant"], axis=1)

        categorical_colums = X.select_dtypes(include=['object']).columns
        numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns

        data_encoded = pd.get_dummies(X, columns=categorical_colums)

        scaled_data = data_encoded.copy()

        # SCALE DATA
        print("<<< INFO: Scaling numerical transformed data")
        col_names = numerical_ix
        features = scaled_data[col_names]
        scaler = MinMaxScaler().fit(features.values)
        features = scaler.transform(features.values)
        scaled_data[col_names] = features
        print("<<< INFO: Scaling done")
        data_project = pd.merge(scaled_data, identification_data, on="MutantID")

        print(data_encoded.head(10))

        data_project = pd.merge(data_project, Y, on="MutantID")

        logging.info(data_project.shape)
        print("\n\n")
        logging.info(data_project.columns)
        print("\n\n")
        logging.info(data_project.dtypes)
        print("\n\n")
        print(data_project.describe())
        print("\n\n")
        print(data_project['Relevant'].value_counts())
        print("\n\n")
        print(data_project['Minimal'].value_counts())

        data_project.to_csv(output_path + "/" + "extracted_mutants_metrices_preprocessed_scaled_" + project.split("-")[-1] + "_hard.csv", header=True)

    # logging.info(data_encoded.shape)
    # print("\n\n")
    # logging.info(data_encoded.columns)
    # print("\n\n")
    # logging.info(data_encoded.dtypes)
    #
    # pca = PCA()
    # pca.fit(data_encoded)
    # cumsum = np.cumsum(pca.explained_variance_ratio_)
    # d = np.argmax(cumsum >= 0.95) + 1
    #
    # print()
    # print("Number of dimensions")
    # print(d)
    #
    # plt.figure(figsize=(6, 4))
    # plt.plot(cumsum, linewidth=3)
    # plt.axis([0, 400, 0, 1])
    # plt.xlabel("Dimensions")
    # plt.ylabel("Explained Variance")
    # plt.plot([d, d], [0, 0.95], "k:")
    # plt.plot([0, d], [0.95, 0.95], "k:")
    # plt.plot(d, 0.95, "ko")
    # plt.annotate("Elbow", xy=(65, 0.85), xytext=(70, 0.7),
    #              arrowprops=dict(arrowstyle="->"), fontsize=16)
    # plt.grid(True)
    # save_fig("explained_variance_plot")
    # plt.show()

    # pca = PCA(n_components=0.95)
    # X_reduced = pca.fit_transform(data_encoded)
    #
    # print(pca.n_components_)
    # print(np.sum(pca.explained_variance_ratio_))

def inter_validation(data,
                     target_class,
                     training_threadshold,
                     evaluation_threadshold):
    target_data = target_class + ["CommitDate"]
    Y = data[target_data]
    X = data.drop(["Minimal", "Relevant"], axis=1)

    unique_commits = data["CommitDate"].unique()
    logging.info("Number of unique commits: {number}".format(number=len(unique_commits)))
    commit_threashold_for_traning = int(len(unique_commits) * training_threadshold)
    commit_threashold_for_evaluation = int(len(unique_commits) * evaluation_threadshold)
    logging.info(
        "Commit index as training threadshold: {number}".format(number=unique_commits[commit_threashold_for_traning]))
    logging.info("Commit index as evoluation threadshold: {number}".format(
        number=unique_commits[commit_threashold_for_evaluation]))

    train_commit = unique_commits[commit_threashold_for_traning]
    eval_commit = unique_commits[commit_threashold_for_evaluation]

    X_train, X_test, X_eval = X[X["CommitDate"] < train_commit], X[X["CommitDate"] > eval_commit], X[
        (X["CommitDate"] >= train_commit) & (X["CommitDate"] <= eval_commit)]
    y_train, y_test, y_eval = Y[Y["CommitDate"] < train_commit], Y[Y["CommitDate"] > eval_commit], Y[
        (Y["CommitDate"] >= train_commit) & (X["CommitDate"] <= eval_commit)]

    # y_train_size = len(X_train)
    # y_train, y_test = Y.iloc[:, :y_train_size], Y.iloc[: ,y_train_size:]

    X_train = X_train.drop(["CommitDate"], axis=1)
    X_test = X_test.drop(["CommitDate"], axis=1)
    y_train = y_train.drop(["CommitDate"], axis=1)
    y_test = y_test.drop(["CommitDate"], axis=1)
    X_eval = X_eval.drop(["CommitDate"], axis=1)
    y_eval = y_eval.drop(["CommitDate"], axis=1)

    logging.info('Number of rows in Train dataset: {X_train.shape[0]}')
    logging.info(y_train[target_class[0]].value_counts())
    logging.info('Number of rows in Test dataset: {X_test.shape[0]}')
    logging.info(y_test[target_class[0]].value_counts())
    logging.info('Number of rows in Eval dataset: {x_eval.shape[0]}')
    logging.info(y_eval[target_class[0]].value_counts())

    logging.info("Percentage of training set: {:0.2f}%".format(len(X_train) / len(X)))
    logging.info("Percentage of classes in training set: Positive: {:0.2f}% / Negative: {:0.2f}%".format(
        y_train[target_class[0]].value_counts()[1] / len(y_train),
        y_train[target_class[0]].value_counts()[0] / len(y_train)))
    print("\n")

    negative_instances = len(y_test[y_test[target_class[0]] == 0])
    positive_instances = len(y_test[y_test[target_class[0]] == 1])

    logging.info("Percentage of testing set: {:0.2f}%".format(len(X_test) / len(X)))
    groundtruth_proportion_of_not_relevant = negative_instances / len(y_test)
    groundtruth_proportion_of_relevant = positive_instances / len(y_test)
    logging.info("Percentage of classes in testing set: Positive: {:0.2f}% / Negative: {:0.2f}%".format(
        groundtruth_proportion_of_relevant, groundtruth_proportion_of_not_relevant))
    logging.info("Test set size: {}".format(len(X_test)))
    logging.info("Relevant mutants in test set: {}".format(y_test[target_class[0]].value_counts()[1]))
    print("\n")

    logging.info("Percentage of validation set: {:0.2f}%".format(len(X_eval) / len(X)))
    logging.info("Percentage of classes in validation set: Positive: {:0.2f}% / Negative: {:0.2f}%".format(
        y_eval[target_class[0]].value_counts()[1] / len(y_eval),
        y_eval[target_class[0]].value_counts()[0] / len(y_eval)))

    return X_train, X_test, y_train, y_test, X_eval, y_eval

def split_data(data,
               project_name,
               target_class,
               training_threadshold,
               evaluation_threadshold,
               validation_type):
    data = data.sort_values(by="CommitDate", ascending=True)

    # if project_name is not None:
    #     data = data[data["Project"] == project_name]
    identification_features = ["Project", "Mutator",
                               "SourceFile", "MutatedClass", "MutatedMethod", "LineNumber", "Index", "Block",
                               "MethodDescription"]
    identification_data = data[identification_features]
    data = data.drop(identification_features, axis=1)
    if validation_type in ["Intra", "Inter", "Inter_c"]:
        X_train, X_test, y_train, y_test, X_eval, y_eval = inter_validation(data=data,
                                                                                target_class=target_class,
                                                                                training_threadshold=training_threadshold,
                                                                                evaluation_threadshold=evaluation_threadshold)

    return identification_data, X_train, X_test, y_train, y_test, X_eval, y_eval


if __name__ == '__main__':
    arguments = get_arguments()

    path_to_training_data, path_to_model = initialise_usage_directories(arguments.working_directory, make_time_dirs=True)

    # Load DATA
    data = load_data(path_to_file=arguments.path_to_file)

    data_preparation(data=data, output_path=path_to_training_data)

    # Split data -> # Check if split by project, # Check what is target label ["Minimal", "Relevant"], # Check for split ratio, # Check for validation
    # identification_data, X_train, X_test, y_train, y_test, X_eval, y_eval = split_data(data=data,
    #            project_name=arguments.project_name,
    #            target_class=["Relevant"],
    #            training_threadshold=0.7,
    #            evaluation_threadshold=0.8,
    #            validation_type="Intra")
    #
    # not_rel = len(data[data['Relevant'] == 0])
    # rel = len(data[data['Relevant'] == 1])
    #
    # # params_grid = {
    # #     'max_depth': [4, 6, 8, 10],
    # #     'n_estimators': [50, 100, 200, 400, 600],
    # #     'learning_rate': [0.01, 0.03, 0.05],
    # #     'min_child_weight' : [1],
    # #     'gamma' : [0],
    # #     'subsample' : [0.8],
    # #     'colsample_bytree' : [1],
    # #     'scale_pos_weight' :  [1]
    # # }
    # #
    # param_test1 = {
    #     'n_estimators': [400, 800, 1200, 2000],
    #     'max_depth': [8, 10, 12],
    #     # 'min_child_weight': range(1, 6, 2)
    #     # 'gamma': [i / 10.0 for i in range(0, 5)]
    #     'learning_rate': [0.01, 0.03, 0.05],
    #     'subsample': [0.33, 0.5, 0.7],
    #     'gamma': [0.25, 0.5, 0.75]
    # }
    #
    # stKfold = StratifiedKFold(n_splits=10, shuffle=False)
    # clf = xgb.XGBClassifier(objective="binary:logistic", tree_method='gpu_hist', sampling_method='gradient_based',
    #                         min_child_weight=4, scale_pos_weight=not_rel/rel)
    # # grid search
    # grid = GridSearchCV(estimator=clf, return_train_score=True, param_grid=param_test1, cv=stKfold,
    #                     scoring=['roc_auc', "f1"], refit="f1", verbose=2)
    # # grid = RandomizedSearchCV(estimator=clf, param_distributions=params_grid, cv=stKfold,
    # #                     scoring=['roc_auc', "f1"], refit="f1", verbose=2)
    #
    # fit_params = {
    #     "early_stopping_rounds": 100,
    #     "eval_metric": "aucpr",
    #     "eval_set": [[X_eval, y_eval.values.ravel()]]}
    #
    # grid_result = grid.fit(X_train, y_train.values.ravel(), **fit_params)
    #
    # print(grid_result.best_score_)
    # logging.info("Best score {bestscore} for params {bestparams}".format(bestscore=grid_result.best_score_,
    #                                                                      bestparams=grid_result.best_params_))
    #
    # xg_classifier = grid_result.best_estimator_
    #
    # y_probabilities = xg_classifier.predict_proba(X_test)
    #
    # auc_score_test = roc_auc_score(y_test, y_probabilities[:, 1])
    # logging.info("<<< INFO: AUC score: {}".format(auc_score_test))
    # y_prediction = xg_classifier.predict(X_test)
    # f1 = f1_score(y_test, y_prediction)
    # precision = precision_score(y_test, y_prediction)
    # recall = recall_score(y_test, y_prediction)
    # logging.info("<<< INFO: F1 {:0.2f}".format(f1))
    # logging.info("<<< INFO: Precision {:0.2f}".format(precision))
    # logging.info("<<< INFO: Recall {:0.2f}".format(recall))
    # mcc_score = matthews_corrcoef(y_test, y_prediction)
    # logging.info("<<< INFO: MCC score test {:0.2f}".format(mcc_score))
    #
    # results = grid_result.best_estimator_.evals_result()
    #
    # # epochs = len(results['validation_0']['f1'])
    # epochs = len(results['validation_0']['aucpr'])
    # x_axis = range(0, epochs)
    # # plot log loss
    # # fig, ax = plt.subplots()
    # # ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    # # ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    # # ax.legend()
    # # plt.ylabel('Log Loss')
    # # plt.title('XGBoost Log Loss')
    # # plt.savefig(os.path.join(path_to_model,
    # #                          "Log_los:{}:{}.pdf".format('Validation', "curve")),
    # #             format='pdf',
    # #             dpi=1200)
    # # # plt.tight_layout()
    # # # plt.show()
    # #
    # # plot classification error
    # # fig, ax = plt.subplots()
    # # ax.plot(x_axis, results['validation_0']['f1'], label='Train')
    # # ax.plot(x_axis, results['validation_1']['f1'], label='Test')
    # # ax.legend()
    # # plt.ylabel('Classification Error')
    # # plt.title('XGBoost Classification Error')
    # # plt.savefig(os.path.join(path_to_model,
    # #                          "Classification_los:{}:{}.pdf".format('Validation', "curve")),
    # #             format='pdf',
    # #             dpi=1200)
    # # plt.tight_layout()
    # # plt.show()
    #
    # # plot auc loss
    # fig, ax = plt.subplots()
    # ax.plot(x_axis, results['validation_0']['aucpr'], label='Train')
    # ax.plot(x_axis, results['validation_1']['aucpr'], label='Test')
    # ax.legend()
    # plt.ylabel('AUC Loss')
    # plt.title('XGBoost AUC Loss')
    # plt.savefig(os.path.join(path_to_model,
    #                          "AUC_los:{}:{}.pdf".format('Validation', "curve")),
    #             format='pdf',
    #             dpi=1200)
    # # plt.tight_layout()
    # # plt.show()