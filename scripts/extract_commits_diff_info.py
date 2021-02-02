import argparse
from pydriller import RepositoryMining
from pydriller.metrics.process.hunks_count import HunksCount

import os


# import logging
# logging.basicConfig(level=logging.INFO)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract metrices related to the particular commit changes")
    parser.add_argument("project_git_url", type=str, help="Target project git url")
    parser.add_argument("commit_id", type=str, help="Commit on which to perform analysis")

    arguments = parser.parse_args()

    parse_commit_diff(arguments.project_git_url, arguments.commit_id)
