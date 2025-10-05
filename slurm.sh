#! /bin/sh

#SBATCH --job-name=sae_insights # job name
#SBATCH --output=/home/yandex/APDL2425a/group_12/gorodissky/sae/slurm.out # redirect stdout
#SBATCH --error=/home/yandex/APDL2425a/group_12/gorodissky/sae/slurm.err # redirect stderr
#SBATCH --partition=gpu-h100-killable # (see resources section)
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=16 # number of processes
#SBATCH --mem=500000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total
#SBATCH --exclude=t-100 # exclude nodes from the job


python main.py