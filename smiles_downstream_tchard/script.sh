#!/bin/bash

#SBATCH --job-name="generate_few_shots_RN"

#SBATCH --output="generate_few_shots_RN.%j.%N.out"

#SBATCH --partition=gpu-shared

#SBATCH --nodes=1

#SBATCH --gpus=3

#SBATCH --ntasks-per-node=10

#SBATCH --mem=93G

#SBATCH --account=nji102

#SBATCH --no-requeue

#SBATCH -t 12:00:00


python generate_few_shots_data.py -s 0 -p 0.5
python generate_few_shots_data.py -s 1 -p 0.5
python generate_few_shots_data.py -s 2 -p 0.5
python generate_few_shots_data.py -s 3 -p 0.5
python generate_few_shots_data.py -s 4 -p 0.5

python mlp.py --dataset 0
python mlp.py --dataset 1
python mlp.py --dataset 2
python mlp.py --dataset 3
python mlp.py --dataset 4