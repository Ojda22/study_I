#!/bin/bash

project_git_path="$1"
project_bugs_info="$2"
path_to_project_bags_repo="$3"

scriptDirectory="$(pwd)"

while IFS=, read -r col1_bugID col2_tests col3_gitHash col4_date
do
   bug_dir="$col1_bugID"f-dev-pitReports
   echo "$bug_dir"
   path_to_bug="$path_to_project_bags_repo"/"$bug_dir"
   cd "$path_to_bug"
   echo "$(pwd)"

   outputFile="$path_to_bug"/"$col3_gitHash"_changes.json # file to store map<filePath,lines> from git diff
   outputFile_change_files="$path_to_bug"/"$col3_gitHash"_changed_files.txt # file to store list<filesChanged> 

   python3 "$scriptDirectory"/extract_changed_lines_4commit.py "$project_git_path" "$col3_gitHash" "$outputFile" "$outputFile_change_files"
   # python3 /Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/experiment_RM/experiment_icse_scripts/extractChangeMetrices.py $project_git_path $col3_gitHash "$path_to_bug"/mutations.xml

   # java -jar /Users/milos.ojdanic/phd_workspace/Mutants_CI/relevantMutant_Milos/experiment_RM/experiment_icse_scripts/mutast.jar $project_working_version $path_to_project_bags_repo $bug_dir "$col4_date" $project_previous_version

   echo "$col3_gitHash","$col4_date" >> "$path_to_project_bags_repo"/"cli_faults_commits.txt"

   echo "I got column1: $col1_bugID"
   echo "I got column3: $col3_gitHash"
   echo "I got column4: $col4_date"
   echo ""
done < $project_bugs_info

echo "<<< Project name: " $project_git_path
echo "<<< Path to project bags repo: " $path_to_project_bags_repo