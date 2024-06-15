import pandas as pd
import torch
import ast
from tqdm import tqdm

tqdm.pandas()

train_df = pd.read_csv("encoded_train_NA_0.csv")
test_df = pd.read_csv("encoded_test_NA_0.csv")

train_X = train_df.copy()

train_y = train_X.pop('binder')

test_X = test_df.copy()

test_y = test_X.pop('binder')


import numpy as np

from sklearn import svm



classifier = svm.SVC(kernel='linear',verbose=True)

classifier.fit(train_X,train_y)

pred_y = classifier.predict(test_X)

print(pred_y.shape)

from sklearn.metrics import accuracy_score



"""

# Calculate the accuracy of the model

accuracy = accuracy_score(test_y,y_pred)

print(f'Accuracy: {accuracy}')

orig_df_test = pd.read_csv("tchard-data/filtered_data_NA/test-0.csv")

orig_df_test['prediction'] = y_pred

orig_df_test.to_csv('predictions_0',index=False)

"""




from sklearn.metrics import f1_score, roc_auc_score,precision_score,recall_score

auc = roc_auc_score(test_y, pred_y)
print("AUC: ",auc)

pred_y = pred_y>0.5

acc = (test_y==pred_y).sum()/pred_y.shape[0]
print("ACC: ",acc)


f1 = f1_score(test_y, pred_y)
print("F1: ",f1)


precision = precision_score(test_y,pred_y)
print("Precision: ",precision)

recall = recall_score(test_y,pred_y)
print("Recall: ",recall)