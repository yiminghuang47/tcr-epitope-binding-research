#!/bin/bash



#SBATCH --job-name="protbert"



#SBATCH --output="protbert.%j.%N.out"



#SBATCH --partition=gpu-shared



#SBATCH --nodes=1



#SBATCH --gpus=1



#SBATCH --ntasks-per-node=1



#SBATCH --mem=96G



#SBATCH --account=nji102



#SBATCH --no-requeue



#SBATCH -t 00:15:00


python3.9 transformers/examples/pytorch/language-modeling/run_mlm.py  --model_name_or_path Rostlab/prot_bert  --train_file tchard-data/Full_CDR_only/train_dataset_spaced.csv  --validation_file tchard-data/Full_CDR_only/test_dataset_spaced.csv  --per_device_train_batch_size 4  --per_device_eval_batch_size 4  --do_train --do_eval  --output_dir results_protbert --overwrite_output_dir 



