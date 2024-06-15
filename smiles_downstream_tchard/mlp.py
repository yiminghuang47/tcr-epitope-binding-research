import numpy as np

import tensorflow as tf

from tensorflow.keras import layers, models

from tensorflow import keras

import pandas as pd


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset")
args = parser.parse_args()

train_df = pd.read_csv(f"RN_few_shots_data/50_percent/train-{args.dataset}.csv")
test_df = pd.read_csv(f"RN_few_shots_data/50_percent/test-{args.dataset}.csv")
train_X = train_df.copy()
train_y = train_X.pop('binder')
test_X = test_df.copy()
test_y = test_X.pop('binder')



input_shape = (2301,)  # Input shape for each sample in X_train
model = keras.Sequential([
    layers.BatchNormalization(input_shape=input_shape),
    layers.Dense(1024, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),
])

from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=1)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy','AUC','Precision','Recall']
)



num_epochs = 10

batch_size = 128

history = model.fit(train_X, train_y, validation_data=(test_X,test_y), epochs=num_epochs, batch_size=batch_size)
#history = model.fit(train_X, train_y, validation_data=(test_X,test_y), epochs=num_epochs, batch_size=batch_size,verbose=1)
#history = model.fit(train_X, train_y, epochs=num_epochs, batch_size=batch_size)

metrics = model.evaluate(test_X, test_y)

#print(model.metric_names)
print(f'dataset: {args.dataset}')

print(metrics)
print(model.metrics_names)


from sklearn.metrics import f1_score, roc_auc_score,precision_score,recall_score

pred_y = model.predict(test_X).squeeze()

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


history_df = pd.DataFrame(history.history)
print(history_df)