#!/bin/bash

statistics_data="$1"

scriptDirectory="$(pwd)"
echo "<<< Script directory: " $scriptDirectory

{
    read
    while IFS=, read -r commit commit_date fom_mutants mutants_on_change som_mutants mutants_gran relevant_mutants not_relevant_mutants minimal_mutants equivalent minimal_relevant_mutants MS total_tests relevant_tests project project_url
    do

    python3 "$scriptDirectory"/extract_commits_diff_info.py "$project_url" "$commit"

    echo "I got column1: $commit"
    echo "I got column3: $project_url"
    echo ""
    done 
} < $statistics_data
