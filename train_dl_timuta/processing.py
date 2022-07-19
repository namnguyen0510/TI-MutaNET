import os
import pandas as pd
import numpy as np
from utils import *


dirc = 'paad_qcmg_uq_2016'
mutation = pd.read_csv('qcmg_mutation.csv').drop(columns = ['Oncotree Code', 'TMB (nonsynonymous)'])
mutation.columns = ['PATIENT_ID', 'Mutation_Profile']
idx = mutation['PATIENT_ID'].tolist()
clinical = pd.read_csv(os.path.join(dirc, 'data_clinical_patient.txt'), sep = ',').iloc[3:,:].reset_index(drop = True)[['PATIENT_ID','HISTOLOGICAL_SUBTYPE']]
clinical = select_patient(clinical,idx)

mRNA = pd.read_csv(os.path.join(dirc, 'data_RNA_Seq_v2_expression_median.txt'), sep = '\t').drop(columns = 'Entrez_Gene_Id')
gene = mRNA['Hugo_Symbol']
gene = ['PATIENT_ID'] + gene.tolist()
mRNA = mRNA.T.iloc[1:,:].reset_index()
mRNA.columns = gene
idx = mRNA['PATIENT_ID'].tolist()
mutation = select_patient(mutation,idx)
clinical = select_patient(clinical,idx)
idx = mutation['PATIENT_ID'].tolist()
mRNA = select_patient(mRNA,idx)

label = []
for d in clinical['HISTOLOGICAL_SUBTYPE']:
    if d == 'Intraductal Papillary Mucinous Neoplasm with invasion':
        label.append('IPMN')
    else:
        label.append("PAD")
clinical['HISTOLOGICAL_SUBTYPE'] = label

print(mutation)
print(clinical)
print(mRNA)

print(np.unique(clinical['HISTOLOGICAL_SUBTYPE'], return_counts = True))


clinical['Mutation_Profile'] = mutation['Mutation_Profile']
df = pd.concat([clinical, mRNA.drop(columns = ['PATIENT_ID'])], axis = 1).dropna(axis = 1)

print(df)
df.to_csv('processed_qcmg.csv', index = False)








































#
