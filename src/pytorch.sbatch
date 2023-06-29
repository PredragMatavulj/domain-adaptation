#!/bin/bash
#SBATCH --job-name=TL
#SBATCH --partition=main
#SBATCH --nodes=1
###SBATCH --nodelist=n05,n06
###SBATCH --exclude=n01
###SBATCH --gres=gpu:1
#SBATCH --time=500:00:00 #the maximum walltime in format h:m:s
#SBATCH --output slurm.%J.out # STDOUT
#SBATCH --error slurm.%J.err # STDERR
#SBATCH --export=ALL

module load cuda/11.2
module load python/anaconda3 

env NCCL_LL_THRESHOLD=0 /storage/opt/anaconda3/bin/python crossvalidate_TE.py -i $MASTER  
