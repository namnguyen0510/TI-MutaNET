import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models, util
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import *
import xgboost as xgb
import torch
from model import *
from engine import *
from focal_loss import *
import tqdm
from utils import *

def main(masked_gene,study_dirc,embedding,
        seed,
        num_iters,
        num_epochs,
        bow_threshold,
        gene_threshold,
        hid_dim,
        out_dim,
        n_hid_layers,
        dropout,
        lr,
        alpha,
        gamma,
        topk,
        device):

    checkpoint_dirc = '{}/gene_{}__embedding_{}__seed_{}__bt_{}__gt_{}__hd_{}__od_{}__nlayers_{}__dropout_{}__lr_{}__alpha_{}__gamma_{}'.format(study_dirc,masked_gene,
                        embedding, seed, bow_threshold, gene_threshold,
                        hid_dim, out_dim, n_hid_layers, dropout, lr, alpha, gamma)
    try:
        os.mkdir(checkpoint_dirc)
        os.mkdir(os.path.join(checkpoint_dirc,'model'))
    except:
        pass

    ## LOAD DATASET
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_csv('processed_qcmg.csv')
    df = masking_df_by_gene(df, masked_gene)
    targets = np.array(df['HISTOLOGICAL_SUBTYPE'])
    targets[targets == 'IPMN'] = 0
    targets[targets == 'PAD'] = 1
    ## MUTATION PROFILE
    embedding = SentenceTransformer('../embeddings/{}'.format(embedding))
    vectorizer = CountVectorizer()
    muta = df['Mutation_Profile']
    tokenizing = vectorizer.fit(muta)
    muta_bert = embedding.encode(muta)
    muta_bow = tokenizing.transform(muta).toarray()
    selector = VarianceThreshold(threshold=bow_threshold)
    muta_bow = selector.fit_transform(muta_bow)

    ## mRNA
    mRNA = df.iloc[:,3:]
    try:
        mRNA = mRNA.drop(columns = masked_gene)
    except:
        pass
    mRNA = (mRNA - mRNA.min())/(mRNA.max() - mRNA.min())
    selector = VarianceThreshold(threshold = gene_threshold)
    x = selector.fit_transform(mRNA)
    selected_gene = selector.get_feature_names_out(mRNA.columns)
    #print(gene_threshold, len(selected_gene))

    ## AGGREGATE FEATURES
    features = np.concatenate((muta_bert,muta_bow,x), axis=1)
    AUC = []
    ACC = []
    for iter in tqdm.tqdm(range(num_iters)):
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, stratify = targets)
        ## PREPARE TRAIN/TEST QUEUE
        #print("X_train: {}, X_test: {}".format(x_train.shape, x_test.shape))
        x_train = torch.tensor(x_train).float().to(device)
        x_test = torch.tensor(x_test).float().to(device)
        y_train = torch.tensor(y_train.astype(np.int32)).float().reshape(-1,1).to(device)
        y_test = torch.tensor(y_test.astype(np.int32)).float().reshape(-1,1).to(device)

        ## DEFINE MODEL
        model = TIMutaNet(in_dim = x_train.size(1), hid_dim = hid_dim, out_dim = out_dim, n_hid_layers = n_hid_layers, dropout = dropout, device = device).to(device)
        criterion = BinaryFocalLossWithLogits(alpha = alpha, gamma = gamma, reduction = 'mean')
        optimizer = torch.optim.SGD(model.parameters(), lr = lr)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min = 1e-6)

        ## TRAINING MODEL
        best_loss = np.inf
        for epoch in range(num_epochs):
            train_loss, train_acc = train(model, x_train, y_train, criterion, optimizer, schedule)
            test_loss, test_acc = test(model, x_test, y_test, criterion)
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), '{}/model/best_model_{}.pth'.format(checkpoint_dirc,iter))

        ## EVALUATE
        model.load_state_dict(torch.load('{}/model/best_model_{}.pth'.format(checkpoint_dirc,iter)))
        y_pred = model(x_test).detach().cpu().numpy()
        y_test = y_test.detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        AUC.append(roc_auc_score(y_test,y_pred))
        ACC.append(accuracy_score(y_test,y_pred))
        #print(confusion_matrix(y_test,y_pred))

    #print(np.array(AUC).max())
    #print(np.array(ACC).max())
    df = pd.DataFrame([])
    df['AUC'] = AUC
    df['ACC'] = ACC
    df.to_csv('{}/result.csv'.format(checkpoint_dirc), sep = '\t')

    ## SELECT TOP-K MODELS
    AUC = sorted(AUC)[-topk:]
    ACC = sorted(ACC)[-topk:]
    best_auc = np.array(AUC).mean()
    best_acc = np.array(ACC).mean()

    return best_acc, best_auc
