import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd


def select_patient(df, idx):
    mask = []
    for i in range(len(df)):
        if df['PATIENT_ID'].tolist()[i] in idx:
            mask.append(True)
        else:
            mask.append(False)
    df = df[mask].reset_index(drop=True)
    return df


def read_MP(x):
    gene = []
    muta = []
    codon = []
    for i in range(len(x)):
        if i % 3 == 0:
            gene.append(x[i])
        if i % 3 == 1:
            muta.append(x[i])
        if i % 3 == 2:
            codon.append(x[i])
    return gene #, muta, codon

def get_gene_counts(df, filename = None, sorted = True):
    y = [np.array(x, dtype = object).flatten() for x in df]
    y = np.concatenate(y)
    gene, counts = np.unique(y, return_counts=True)
    data = pd.DataFrame([])
    data['Gene'] = gene
    data['Counts'] = counts
    if sorted:
        data = data.sort_values(by = ['Counts'], ascending = False).reset_index(drop=True)
    if filename is not None:
        data.to_csv(filename, index = False)
    return data

def masking_MP_by_gene(m, gene):
    x = []
    m = m.split(' ')
    if gene in m:
        try:
            gene_index = m.index(gene)
            m.pop(gene_index)
            m.pop(gene_index+1)
            m.pop(gene_index+2)
            m = ' '.join(m)
        except:
            print('Gene Not Excluded')
            m = m
            pass
    else:
        m = m
    return m


def masking_df_by_gene(df,gene):
    df = df.drop(columns = gene)
    muta = [masking_MP_by_gene(x,gene) for x in df['Mutation_Profile']]
    return df

























#
