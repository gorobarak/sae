#! /bin/sh

#SBATCH --job-name=barak # job name
#SBATCH --output=/home/yandex/APDL2425a/group_12/gorodissky/sae/%j.log # redirect stdout
#SBATCH --partition=gpu-h100-killable # (see resources section)
#SBATCH --time=24:00:00 # hours:minutes:seconds
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=48G # memory per node
#SBATCH --cpus-per-task=8 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --exclude=t-100 # exclude nodes from the job


script=$1
echo "Running script: $script"
python "$script"