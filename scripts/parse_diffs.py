import argparse
from datetime import datetime

from pydriller import RepositoryMining
import pandas as pd
import os
import sys

sys.path.append(os.getcwd())


def parseCommit(project_git_url, commits):
    total_number_commits = len(commits)
    change_test_contract = 0
    goodCommits = []
    for commit_id in commits:
        print(commit_id)
        goodCommit = True
        for commit in RepositoryMining(path_to_repo=project_git_url, single=commit_id).traverse_commits():
            for m in commit.modifications:
                if m.new_path is not None:
                    if "test" in m.new_path or "Test" in m.new_path:
                        change_test_contract += 1
                        goodCommit = False
                        break
                if m.old_path is not None:
                    if "test" in m.old_path or "Test" in m.old_path:
                        change_test_contract += 1
                        goodCommit = False
                        break
        if goodCommit:
            goodCommits.append(commit_id)

    avarage = change_test_contract / total_number_commits
    print(f"From {total_number_commits} commits: Contract changed by {change_test_contract} which is: {avarage:.3f}")
    return goodCommits


def mineCommits(project_git_url):
    goodCommits = list()
    counter = 0
    ignore_commits_list = ["javadoc", "checkstyle", "import", "tags", "tag", "spelling", "typo", "typos", "cleanup",
                           "syntax", "arranging", "spaces", "a note", "style", "annotation", "documentation",
                           "comments", "comment", "final", "generics", "override "]

    for commit in RepositoryMining(path_to_repo=project_git_url, since=datetime(2005, 1, 1, 0, 0, 0),
                                   only_no_merge=False,
                                   only_modifications_with_file_types=[".java"]).traverse_commits():
        good_commit = False
        # all_tests = True
        # Ignore commits about the javadoc or checkstyle (e.g. "update javadoc" or "make checkstyle happy")
        if any(element in commit.msg.lower() for element in ignore_commits_list):
            continue

        counter += 1
        for modification in commit.modifications:
            # If the commit modifies a something else than the code (e.g. config file or pom file), ignore it
            # If the commit modifies the tests, ignore it
            # if not modification.filename.endswith(".java"):
            #     good_commit = False
            #     break
            if modification.new_path is not None:
                if "test" in modification.new_path or "Test" in modification.new_path:
                    good_commit = True
                    break
            if modification.old_path is not None:
                if "test" in modification.old_path or "Test" in modification.old_path:
                    good_commit = True
                    break

        if not good_commit:
            goodCommits.append(commit.hash + "," + commit.committer_date.strftime("%m/%d/%Y - %H:%M:%S"))

    return goodCommits, counter


def parse(pathOfFile):
    # src/main/java/org/apache/commons/collections4/map/CompositeMap.java
    return pathOfFile.split(".")[0].split("java/", 1)[1].replace("/", ".")


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Finding changed lines")
    # parser.add_argument("project_git_url", type=str, help="URL of the git project that needs to be mined")
    # parser.add_argument("output_file", type=str, help="File where to write mined commits", required=True)
    # parser.add_argument("commit_file_path", type=str, help="Commit which need to be parsed")
    # arguments = parser.parse_args()

    for project in ['collections', 'csv', 'io', 'lang', 'text']:
        file_path = f'./{project}Commits.csv'
        print(file_path)
        commits = pd.read_csv(open(file_path)).to_numpy().flatten()
        print("Commits lenght: ", len(commits))
        project_git = f"https://github.com/apache/commons-{project}.git"
        goodCommits = parseCommit(project_git, commits)
        # goodCommits, counter = mineCommits(arguments.project_git_url)
        output_file_path = f"./contract_intact_{project}.csv"
        with open(output_file_path, "w") as output_file:
            output_file.write("{}\n".format(project_git))
            for commit in goodCommits:
                output_file.write("{}\n".format(commit))
