#!/bin/bash

#SBATCH --job-name="vibtcr_NA"

#SBATCH --output="vibtcr_NA.%j.%N.out"

#SBATCH --partition=gpu-shared

#SBATCH --nodes=1

#SBATCH --gpus=3

#SBATCH --ntasks-per-node=1

#SBATCH --mem=93G

#SBATCH --account=nji102

#SBATCH --no-requeue

#SBATCH -t 12:00:00

python main.py --trainfile tchard-data/filtered_data_NA/train-0.csv --testfile tchard-data/filtered_data_NA/test-0.csv --outfile predictions_NA_vibtcr/predictions-0.csv --epochs 500
python main.py --trainfile tchard-data/filtered_data_NA/train-1.csv --testfile tchard-data/filtered_data_NA/test-1.csv --outfile predictions_NA_vibtcr/predictions-1.csv --epochs 500
python main.py --trainfile tchard-data/filtered_data_NA/train-2.csv --testfile tchard-data/filtered_data_NA/test-2.csv --outfile predictions_NA_vibtcr/predictions-2.csv --epochs 500
python main.py --trainfile tchard-data/filtered_data_NA/train-3.csv --testfile tchard-data/filtered_data_NA/test-3.csv --outfile predictions_NA_vibtcr/predictions-3.csv --epochs 500
python main.py --trainfile tchard-data/filtered_data_NA/train-4.csv --testfile tchard-data/filtered_data_NA/test-4.csv --outfile predictions_NA_vibtcr/predictions-4.csv --epochs 500
