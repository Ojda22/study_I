#!/bin/bash -l

commits_file=$1

broken=0
commits=0

echo "$(pwd)"

function throw_error () {
	# arg1 -> checkout status
	# arg2 -> error message
	# arg3 -> for which commit
	# arg4 -> script directory
	if [[ "$1" != 0 ]]; then
		echo "Commit broken: $2"
		broken=$((broken+1))
	fi
}

while read commit; do
  commits=$((commits + 1))
  echo "Commits: $commits"
  echo "Commit: $commit"
  echo "Commit broken: $broken"

  git clean -fd # clean dir from new files
  git checkout -f $commit^
  mvn clean test -Drat.skip=true
  throw_error "$?" "$commit"

done < "$commits_file"

