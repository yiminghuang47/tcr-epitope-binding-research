#!/bin/bash

#SBATCH --job-name="mlp_few_shots_RN"

#SBATCH --output="mlp_few_shots_RN.%j.%N.out"

#SBATCH --partition=gpu-shared

#SBATCH --nodes=1

#SBATCH --gpus=3

#SBATCH --ntasks-per-node=10

#SBATCH --mem=93G

#SBATCH --account=nji102

#SBATCH --no-requeue

#SBATCH -t 00:10:00
 


python mlp.py --dataset 2