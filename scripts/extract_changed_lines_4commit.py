import argparse
from pydriller import RepositoryMining
import json


def parseCommit(project_git_url, commit_id):
    added_lines_per_file = dict()
    changedFiles = []

    for commit in RepositoryMining(path_to_repo=project_git_url, single=commit_id).traverse_commits():
        for m in commit.modifications:
            if "src/main/" in m.new_path:
                added_lines_per_file[m.new_path] = list(map(lambda x: x[0], m.diff_parsed["added"]))
                changedFiles.append(parse(m.new_path))

    print(added_lines_per_file)
    print(changedFiles)
    return added_lines_per_file, changedFiles


def parse(pathOfFile):
    # src/main/java/org/apache/commons/collections4/map/CompositeMap.java
    return pathOfFile.split(".")[0].split("java/", 1)[1].replace("/", ".")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Finding changed lines")
    parser.add_argument("project_git_url", type=str, help="URL of the git project that needs to be mined")
    parser.add_argument("commit_id", type=str, help="Commit which need to be parsed")
    parser.add_argument("output_file", type=str, help="File where to write files with changed lines")
    parser.add_argument("changed_files", type=str, help="File where to write changed files name")

    arguments = parser.parse_args()

    added_lines_per_file, changedFiles = parseCommit(arguments.project_git_url, arguments.commit_id)

    with open(arguments.output_file, "w") as output_file:
        json.dump(added_lines_per_file, output_file, indent=4, sort_keys=True)

    with open(arguments.changed_files, "w") as output_file:
        output_file.write(",".join(changedFiles))
