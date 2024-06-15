import torch

if torch.cuda.is_available():

 dev = "cuda:0"

else:

 dev = "cpu"



device = torch.device(dev)



#import torch_xla

#import torch_xla.core.xla_model as xm

#device = xm.xla_device()

print(device)

from transformers import BertModel, BertTokenizer

from tqdm import tqdm

import pandas as pd

import re

tqdm.pandas()


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
args = parser.parse_args()

def encode_sequence(seq):

  seq = re.sub(r"[UZOB]", "X", seq)

  encoded_seq = tokenizer(seq, return_tensors='pt').to(device)

  output = model(**encoded_seq)

  return output.pooler_output.flatten().tolist()



def add_space(seq):

  return ' '.join(seq)



tokenizer = BertTokenizer.from_pretrained("yiminghuang47/prot_bert-finetuned-tchard", do_lower_case=False )

model = BertModel.from_pretrained("yiminghuang47/prot_bert-finetuned-tchard").to(device)



train_df = pd.read_csv(f"tchard-data/filtered_data_RN/train-{args.dataset}.csv")

test_df = pd.read_csv(f"tchard-data/filtered_data_RN/test-{args.dataset}.csv")





train_df['CDR3a'] = train_df['CDR3a'].progress_apply(add_space)

train_df['CDR3b'] = train_df['CDR3b'].progress_apply(add_space)

train_df['peptide'] = train_df['peptide'].progress_apply(add_space)



train_df['Encoded_CDR3a'] = train_df["CDR3a"].progress_apply(encode_sequence)

train_df['Encoded_CDR3b'] = train_df["CDR3b"].progress_apply(encode_sequence)

train_df['Encoded_peptide'] = train_df["peptide"].progress_apply(encode_sequence)

train_df['Encoded_Input'] = train_df['Encoded_CDR3a'] + train_df['Encoded_CDR3b'] + train_df['Encoded_peptide']


test_df['CDR3a'] = test_df['CDR3a'].progress_apply(add_space)

test_df['CDR3b'] = test_df['CDR3b'].progress_apply(add_space)

test_df['peptide'] = test_df['peptide'].progress_apply(add_space)



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

train_df_encoded.to_csv(f"encoded_train_RN_{args.dataset}.csv",index=False)



test_df_encoded = test_X.copy()

test_df_encoded['binder'] = test_y

test_df_encoded.to_csv(f"encoded_test_RN_{args.dataset}.csv",index=False)
