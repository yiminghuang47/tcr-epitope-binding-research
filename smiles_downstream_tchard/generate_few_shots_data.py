

import os

import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--split",'-s',help="data split #")
parser.add_argument("--percent",'-p',help="percentage of test data to include in the train data")
args =parser.parse_args()

SPLIT = args.split
PERCENT = float(args.percent)


train_df_orig = pd.read_csv(f"tcr-data/filtered_data_RN/train-{SPLIT}.csv")
test_df_orig = pd.read_csv(f"tcr-data/filtered_data_RN/test-{SPLIT}.csv")

train_df_encoded_orig = pd.read_csv(f"encoded_RN/encoded_train_RN_{SPLIT}.csv")
test_df_encoded_orig = pd.read_csv(f"encoded_RN/encoded_test_RN_{SPLIT}.csv")
df_orig = pd.concat([train_df_orig,test_df_orig])


train_df_encoded_new = train_df_encoded_orig.copy()
test_df_encoded_new = pd.DataFrame()

for peptide in test_df_orig['peptide'].unique().tolist():
    all = test_df_orig.loc[test_df_orig['peptide']==peptide]
    train,test = train_test_split(all,train_size=PERCENT,shuffle=True)
    train_df_encoded_new = pd.concat([train_df_encoded_new, test_df_encoded_orig.iloc[train.index]])
    test_df_encoded_new = pd.concat([test_df_encoded_new, test_df_encoded_orig.iloc[test.index]])
    
 

print(train_df_encoded_orig.shape)
print(test_df_encoded_orig.shape)
print(train_df_encoded_new.shape)
print(test_df_encoded_new.shape)


train_df_encoded_new.to_csv(f"RN_few_shots_data/50_percent/train-{SPLIT}.csv",index=False)

test_df_encoded_new.to_csv(f"RN_few_shots_data/50_percent/test-{SPLIT}.csv",index=False)

