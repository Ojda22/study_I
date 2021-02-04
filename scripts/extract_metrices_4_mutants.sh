#!bin/bash

# speficity number of nodes
#SBATCH -N 2

# specify number of tasks/cores per node required
#SBATCH --ntasks-per-node 20

# specify the walltime
#SBATCH --time=2-00:00:00

# set to email at start,end and failed jobs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=milos.ojdanic@uni.lu

exe_path=$1
#scriptDirectory="${SLURM_SUBMIT_DIR}"
commitsDirectory=$(realpath "$2")
projectSourceCode=$(realpath "$3")
previousVersion=$4
echo "Projects directory: " $commitsDirectory
# commits=$(ls -d "$projectsDirectory"/*)
resultsPath=$(realpath "$5")
commitsFile=$6

if [ "$7" == "HPC" ]; then
	scriptDirectory="${SLURM_SUBMIT_DIR}"
	export PATH="/home/users/mojdanic/bin/apache-maven-3.6.3/bin:$PATH"
	module load lang/Java/1.8.0_162
	
	echo "INFO >>> Submit dir. : ${SLURM_SUBMIT_DIR}"
else
	scriptDirectory=$(pwd)
fi

function throw_error () {
	# arg1 -> checkout status
	# arg2 -> error message
	# arg3 -> for which commit
	# arg4 -> script directory
	if [[ "$1" != 0 ]]; then
		echo "ERROR >>> $2: $3"
		cd "$4"
		continue
	fi
}

echo "Script directory: "$scriptDirectory
echo "Exe: "$exe_path

echo "Project mutants directory: "$commitsDirectory
echo "Project source code: "$projectSourceCode
echo "Project metrices results path: "$resultsPath
#echo Project commits: "$commits"
echo "Project for previous version: " $previousVersion
echo "Results path: " $resultsPath
echo ""


# set comma as delimeter
IFS=","

# module load lang/Python/3.7.2-GCCcore-8.2.0
# pip install --user pandas

while read line; do

	echo $line

	#Read the split words into an array based on comma delimiter
	read -a commit_date_array <<< "$line"

	commitID=${commit_date_array[0]}
	echo "COMMIT ID: $commitID"
	commit_date=${commit_date_array[1]}
	echo "COMMIT DATE: $commit_date"

	commit="$commitsDirectory/$commitID"
	echo "COMMIT path: " $commit

    cd $projectSourceCode
    echo "Entering project path: " $projectSourceCode

    echo " "
    echo "Executing >>> checkout and compiling working version"
    echo " "
    git clean -fd
    git checkout .
    git checkout -f $commitID

    throw_error "$?" "checkout problem for commit" $commitID "$scriptDirectory"

    echo "Compile project for commit: " $commitID
    mvn clean compile -Drat.skip=true

    throw_error "$?" "Current version compile problem" $commitID "$scriptDirectory"

    echo " "
    echo "Executing >> checkout and compiling previous version"
    echo " "

    cd $previousVersion
    git clean -fd
    git checkout .
    git checkout -f $commitID^

    throw_error "$?" "checkout problem for commit" $commitID "$scriptDirectory"

    echo "Compile project for previous commit: " $commitID
    mvn clean compile -Drat.skip=true

    throw_error "$?" "Previous version compile problem" $commitID "$scriptDirectory"

    cd $commit
    echo "Entering commit results directory: " $(pwd)
    echo "Commit results path: " $commit

    echo " "
    echo "Executing >>> mutast.jar"
    echo " "

    commitDateTime=$(echo "$commit_date" | tr -d "[:space:]")

    echo "COMMIT DATE TIME: " $commitDateTime

    java -jar "$exe_path/"mutast.jar $projectSourceCode $commitsDirectory $commitID $commitDateTime $previousVersion

    throw_error "$?" "Problem with metrices extraction" $commitID "$scriptDirectory"

    tail -n +2 "$commit/"extractedMetrices_minimalVersion.csv >> "$resultsPath/"EXTRACT-FILE-v1.0.csv

    throw_error "$?" "Problems with file transfering" $commitID "$scriptDirectory"

done < $commitsFile
