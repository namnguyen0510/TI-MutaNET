import os
import numpy as np
import pandas as pd
from utils import *
import tqdm as tqdm
from gensim.models import Word2Vec
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import seaborn as sns
import torch.nn as nn
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models, util
from torch.utils.data import DataLoader
import seaborn as sns
import argparse

parser = argparse.ArgumentParser("TrainEmbedding")
parser.add_argument('--seed', type =int, default = 1, help = 'Set seed for study')
parser.add_argument('--batch_size', type =int, default = 32, help = 'Set batch size')
parser.add_argument('--model', type = str, default = 'ALBERT', help = 'Select MP encoder. Support models: ALBERT, BERT, MiniLM_L6, MiniLM_L12')
parser.add_argument('--output_dim', type = int, default = 64, help = 'Output dim')
parser.add_argument('--max_length', type = int, default = 255, help = 'Maximum sequence length')
parser.add_argument('--epochs', type = int, default = 30, help = 'Number of epochs')
args = parser.parse_args()


# SET SEED AND CREATE STUDY
seed = args.seed
study_name = 'seed_{}__model_{}__outdim_{}'.format(args.seed, args.model, args.output_dim)
try:
    os.mkdir(study_name)
except:
    pass
torch.manual_seed(seed)
np.random.seed(seed)

# LOAD DATASET
df = pd.read_csv('PanCancer_MSK.csv')
df = pd.concat([df[df['Oncotree Code'] == 'IPMN'], df[df['Oncotree Code'] == 'PAAD']])
trainset = df
IPMN = trainset[trainset['Oncotree Code'] == 'IPMN']['Mutation_Profile'].values.tolist()
PAAD = trainset[trainset['Oncotree Code'] == 'PAAD']['Mutation_Profile'].values.tolist()
print("IPMN: {}, PAD: {}".format(len(IPMN),len(PAAD)))

# CREATE POS/NEG PAIRS FOR CONTRASTIVE LEARNING
pos_pair = []
for a in IPMN:
    for b in IPMN:
        pos_pair.append([a,b])
neg_pair = []
for a in IPMN:
    for b in PAAD:
        neg_pair.append([a,b])
print('# Pre-training Patients: {}'.format(len(trainset)))
print("Positive Pairs: {}".format(len(pos_pair)))
print("Negative Pairs: {}".format(len(neg_pair)))
print(len(pos_pair) + len(neg_pair))
pos_pair = [InputExample(texts = p, label = 1.) for p in pos_pair]
neg_pair = [InputExample(texts = p, label = -1.) for p in neg_pair]
trainqueue = DataLoader(pos_pair+neg_pair, shuffle=True, batch_size=args.batch_size)

# LOAD MODEL
if args.model == 'ALBERT':
    embedding_layer = models.Transformer('nreimers/albert-small-v2', max_seq_length = args.max_length)
elif args.model == 'BERT':
    embedding_layer = models.Transformer('bert-base-uncased', max_seq_length = args.max_length)
elif args.model == 'MiniLM_L6':
    embedding_layer = models.Transformer('nreimers/MiniLM-L6-H384-uncased', max_seq_length = args.max_length)
elif args.model == 'MiniLM_L12':
    embedding_layer = models.Transformer('microsoft/MiniLM-L12-H384-uncased', max_seq_length = args.max_length)
else:
    print('Model is Not Implemented!')


pooling_layer = models.Pooling(embedding_layer.get_word_embedding_dimension())
dense_layer = models.Dense(in_features = pooling_layer.get_sentence_embedding_dimension(), out_features = args.output_dim, activation_function=nn.Tanh())
model = SentenceTransformer(modules = [embedding_layer,pooling_layer,dense_layer])
criterion = losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(trainqueue, criterion)], epochs=args.epochs, warmup_steps=100, output_path = '{}/model'.format(study_name),
        checkpoint_path = '{}/checkpoint'.format(study_name), checkpoint_save_steps = int(len(trainqueue)/args.batch_size))






































#
