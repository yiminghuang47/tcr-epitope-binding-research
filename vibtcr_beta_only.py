

import pandas as pd

import torch

import numpy as np

import random






from vibtcr.vibtcr.vibtcr.dataset import TCRDataset

from vibtcr.vibtcr.vibtcr.mvib.mvib import MVIB

from vibtcr.vibtcr.vibtcr.mvib.mvib_trainer import TrainerMVIB

from torch.utils.data.sampler import WeightedRandomSampler

from sklearn.model_selection import train_test_split

from tqdm import tqdm



from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, auc

import pandas as pd

import torch

import argparse

import sys

import os



parser = argparse.ArgumentParser(description='Specify parameters')

parser.add_argument("-tr", "--trainfile",

                    help="Specify the full path of the training file with TCR sequences")

parser.add_argument("-te", "--testfile",

                    help="Specify the full path of the file with TCR sequences")

parser.add_argument("-o", "--outfile", default=sys.stdout,

                    help="Specify output file")

parser.add_argument("-e", "--epochs", default=500, type=int,

                    help="Specify the number of epochs")

args = parser.parse_args()





metrics = [

    'auROC',

    'Accuracy',

    'Recall',

    'Precision',

    'F1 score',

    'auPRC'

]





def pr_auc(y_true, y_prob):

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    pr_auc = auc(recall, precision)

    return pr_auc





def get_scores(y_true, y_prob, y_pred):

    

    


    scores = [

        roc_auc_score(y_true, y_prob),

        accuracy_score(y_true, y_pred),

        recall_score(y_true, y_pred),

        precision_score(y_true, y_pred),

        f1_score(y_true, y_pred),

        pr_auc(y_true, y_prob)

    ]



    df = pd.DataFrame(data={'score': scores, 'metrics': metrics})

    return df





def set_random_seed(random_seed):

    random.seed(random_seed)

    np.random.seed(random_seed)

    torch.manual_seed(random_seed)

    torch.cuda.manual_seed(random_seed)

    torch.cuda.manual_seed_all(random_seed)





device = torch.device('cpu')



batch_size = 4096

epochs = args.epochs

lr = 1e-3



z_dim = 150

early_stopper_patience = 5

monitor = 'auROC'

lr_scheduler_param = 10

joint_posterior = "aoe"



beta = 1e-6





df_train = pd.read_csv(args.trainfile)

df_test = pd.read_csv(args.testfile)



#set_random_seed(0)



scaler = TCRDataset(df_train.copy(), torch.device(

    "cpu"), cdr3b_col='CDR3b', cdr3a_col=None, gt_col='binder').scaler



ds_test = TCRDataset(df_test, torch.device(

    "cpu"), cdr3b_col='CDR3b', cdr3a_col=None, gt_col='binder', scaler=scaler)



df_train, df_val = train_test_split(

    df_train, test_size=0.2, stratify=df_train.binder, random_state=0)



# train loader with balanced sampling

ds_train = TCRDataset(df_train, device, cdr3b_col='CDR3b',

                      cdr3a_col=None, gt_col='binder', scaler=scaler)

class_count = np.array(

    [df_train[df_train.binder == 0].shape[0], df_train[df_train.binder == 1].shape[0]])

weight = 1. / class_count



samples_weight = torch.tensor([weight[int(s)] for s in df_train.binder])

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

train_loader = torch.utils.data.DataLoader(

    ds_train,

    batch_size=batch_size,

    sampler=sampler

)



# val loader with balanced sampling

ds_val = TCRDataset(df_val, device, cdr3b_col='CDR3b',

                    cdr3a_col=None, gt_col='binder', scaler=scaler)

class_count = np.array(

    [df_val[df_val.binder == 0].shape[0], df_val[df_val.binder == 1].shape[0]])

weight = 1. / class_count

samples_weight = torch.tensor([weight[int(s)] for s in df_val.binder])

sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

val_loader = torch.utils.data.DataLoader(

    ds_val,

    batch_size=batch_size,

    sampler=sampler

)



model = MVIB(z_dim=z_dim, device=device,

             joint_posterior=joint_posterior).to(device)



trainer = TrainerMVIB(

    model,

    epochs=epochs,

    lr=lr,

    beta=beta,

    checkpoint_dir=".",

    mode="bimodal",

    lr_scheduler_param=lr_scheduler_param

)

checkpoint = trainer.train(

    train_loader, val_loader, early_stopper_patience, monitor)



# test

model = MVIB.from_checkpoint(checkpoint, torch.device("cpu"))

pred = model.classify(

    pep=ds_test.pep, cdr3b=ds_test.cdr3b, cdr3a=None)

pred = pred.detach().numpy()

df_test['prediction'] = pred.squeeze().tolist()



# save results for further analysis

df_test.to_csv(

    args.outfile,

    index=False

)

