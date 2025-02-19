#! /bin/sh

#SBATCH --job-name=butterfly_comperasion
#SBATCH --output=/home/yandex/APDL2425a/group_12/gorodissky/sae/butterfly_comperasion.out # redirect stdout
#SBATCH --error=/home/yandex/APDL2425a/group_12/gorodissky/sae//butterfly_comperasion.err # redirect stderr
#SBATCH --partition=gpu-h100-killable # (see resources section)
#SBATCH --time=1000 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=4 # number of processes
#SBATCH --mem=500000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=4 # GPUs in total


python main.py