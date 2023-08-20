import numpy as np
import pandas as pd
from einops import rearrange

path = "dataset/multiomics/"

# clinical data
clinical_data_raw = pd.read_csv(path + "1_220830_clinical_data.txt", sep='\t', index_col=1)

# gfre data
gfre_data_raw = pd.read_csv(path + "1_220830_gfre.txt", sep='\t', index_col=0)
gfre_data_raw.index = "g" + gfre_data_raw.index

# protein data
protein_data_raw = pd.read_csv(path + "1_220830_protein.txt", sep='\t', index_col=0)
protein_data_raw.index = "p" + protein_data_raw.index

# rna data
rna_data_raw = pd.read_csv(path + "1_220830_rna.txt", sep='\t', index_col=0)
rna_data_raw.index = "r" + rna_data_raw.index

# sig data
sig_data_raw = pd.read_csv(path + "1_220830_sig.txt", sep='\t', index_col=0)
sig_data_raw.index = "s" + sig_data_raw.index

# Check datasets
gfre_data_raw.describe()
protein_data_raw.describe()
rna_data_raw.describe()
sig_data_raw.describe()

# Normalization
gfre_data_raw=(gfre_data_raw-gfre_data_raw.mean())/gfre_data_raw.std()
protein_data_raw=(protein_data_raw-protein_data_raw.mean())/protein_data_raw.std()
rna_data_raw=(rna_data_raw-rna_data_raw.mean())/rna_data_raw.std()
sig_data_raw=(sig_data_raw-sig_data_raw.mean())/sig_data_raw.std()

# Create dataset
df = pd.concat([gfre_data_raw, protein_data_raw, rna_data_raw, sig_data_raw], join='inner', axis=0)
# normalization
df = (df-df.mean())/df.std()
df.describe()

df = df.T
df.loc[:, 'vital_status'] = clinical_data_raw['vital_status']
data = df.to_numpy()
X = data[:, :-1]
y = data[:, -1]
gene_size = int(X.shape[1]/4)
tmp = np.stack((X[:, :gene_size], X[:, gene_size:2*gene_size], X[:, 2*gene_size:3*gene_size], X[:, 3*gene_size:]))
inputs = rearrange(tmp, 'm b g -> b g m')
gene_input = rearrange(inputs, 'b g m -> g (m b)')

np.savetxt("dataset/multiomics/gene_input.txt", gene_input, delimiter=',')
np.savetxt("dataset/multiomics/patient_label.txt", y.reshape(-1, 1), delimiter=',')

# ========================================================

import numpy as np

x = np.load("dataset/multiomics/input.npy").astype('float32')
y = np.load("dataset/multiomics/label.npy").astype('float32')

x = x[:, :, 3].reshape(105, 14123, 1)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5)

train_set = []
val_set = []
for train_index, val_index in skf.split(x, y):
    x_train, x_val = x[train_index], x[val_index]
    y_train, y_val = y[train_index], y[val_index]
    train_set.append((x_train, y_train))
    val_set.append((x_val, y_val))

train_x = train_set[0][0]
train_y = train_set[0][1]

import os
import torch

num_classes = 2
number_per_class = {}

for i in range(num_classes):
    number_per_class[i] = 0

def custom_datasave(input, label):
    path = 'dataset/sig/train/' + str(label) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(torch.from_numpy(input), path + str(number_per_class[label]) + '.pt')
    number_per_class[label] += 1

def process():
    for i, (input, target) in enumerate(zip(train_x, train_y)):
        print("[ Current Index: " + str(i) + " ]")
        custom_datasave(input, int(target.item()))

process()

val_x = val_set[0][0]
val_y = val_set[0][1]

num_classes = 2
number_per_class = {}

for i in range(num_classes):
    number_per_class[i] = 0

def custom_datasave(input, label):
    path = 'dataset/sig/val/' + str(label) + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save(torch.from_numpy(input), path + str(number_per_class[label]) + '.pt')
    number_per_class[label] += 1

def process():
    for i, (input, target) in enumerate(zip(val_x, val_y)):
        print("[ Current Index: " + str(i) + " ]")
        custom_datasave(input, int(target.item()))

process()