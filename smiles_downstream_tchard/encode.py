import numpy as np
import torch


if torch.cuda.is_available():

 dev = "cuda"

else:

 dev = "cpu"

 

device = torch.device(dev)



#import torch_xla

#import torch_xla.core.xla_model as xm

#device = xm.xla_device()

print(device)

from transformers import AutoModelForMaskedLM, AutoTokenizer

from tqdm import tqdm

import pandas as pd

import re

from rdkit import Chem

tqdm.pandas()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
args = parser.parse_args()


def encode_sequence(seq):
    
  mol = Chem.MolFromSequence(seq)
  smiles = Chem.MolToSmiles(mol)

  #seq = re.sub(r"[UZOB]", "X", seq)

  #encoded_seq = tokenizer(smiles, return_tensors='pt').to(device)
  
  encoded_seq = tokenizer(smiles, padding='max_length', truncation=True,max_length=500,return_tensors='pt').to(device)


  output = model(**encoded_seq)
  logits = output.logits
  logits = logits.detach().cpu().numpy()
  logits = logits.squeeze()
  logits = np.mean(logits,axis=0)
  logits = logits.tolist()

  return logits






tokenizer = AutoTokenizer.from_pretrained("yiminghuang47/ChemBERTa-zinc-base-v1-finetuned-tchard", do_lower_case=False )

model = AutoModelForMaskedLM.from_pretrained("yiminghuang47/ChemBERTa-zinc-base-v1-finetuned-tchard").to(device)



train_df = pd.read_csv(f"tcr-data/filtered_data_RN/train-{args.dataset}.csv")

test_df = pd.read_csv(f"tcr-data/filtered_data_RN/test-{args.dataset}.csv")



train_df['Encoded_CDR3a'] = train_df["CDR3a"].progress_apply(encode_sequence)

train_df['Encoded_CDR3b'] = train_df["CDR3b"].progress_apply(encode_sequence)

train_df['Encoded_peptide'] = train_df["peptide"].progress_apply(encode_sequence)

train_df['Encoded_Input'] = train_df['Encoded_CDR3a'] + train_df['Encoded_CDR3b'] + train_df['Encoded_peptide']




test_df['Encoded_CDR3a'] = test_df["CDR3a"].progress_apply(encode_sequence)

test_df['Encoded_CDR3b'] = test_df["CDR3b"].progress_apply(encode_sequence)

test_df['Encoded_peptide'] = test_df["peptide"].progress_apply(encode_sequence)

test_df['Encoded_Input'] = test_df['Encoded_CDR3a'] + test_df['Encoded_CDR3b'] + test_df['Encoded_peptide']



#test_df.to_csv("encoded_test.csv",index=False)


train_X = pd.DataFrame(train_df['Encoded_Input'].tolist())

train_y = train_df['binder']

test_X = pd.DataFrame(test_df['Encoded_Input'].tolist())

test_y = test_df['binder']

train_df_encoded = train_X.copy()

train_df_encoded['binder'] = train_y

train_df_encoded.to_csv(f"encoded_RN/encoded_train_RN_{args.dataset}.csv",index=False)



test_df_encoded = test_X.copy()

test_df_encoded['binder'] = test_y

test_df_encoded.to_csv(f"encoded_RN/encoded_test_RN_{args.dataset}.csv",index=False)
