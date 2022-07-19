import numpy as np
import pandas as pd
from gensim.models.callbacks import CallbackAny2Vec

def get_TMB(df):
    y = []
    for x in df:
        if x < 0.034:
            y.append(0)
        if x >= 0.034 and x < 0.1:
            y.append(1)
        if x >= 0.1 and x < 4:
            y.append(2)
        if x >= 4 and x < 8:
            y.append(3)
        if x >= 8:
            y.append(4)
    return np.array(y)



def get_muta_words(df):
    df['Variant_Classification'][df['Variant_Classification'] == 'Frame_Shift_Del'] = 'FSD'
    df['Variant_Classification'][df['Variant_Classification'] == 'Frame_Shift_Ins'] = 'FSI'
    df['Variant_Classification'][df['Variant_Classification'] == 'In_Frame_Del'] = 'IFD'
    df['Variant_Classification'][df['Variant_Classification'] == 'In_Frame_Ins'] = 'IFI'
    df['Variant_Classification'][df['Variant_Classification'] == 'Missense_Mutation'] = 'Missense'
    df['Variant_Classification'][df['Variant_Classification'] == 'Nonsense_Mutation'] = 'Nonsense'
    df['Variant_Classification'][df['Variant_Classification'] == 'Nonstop_Mutation'] = 'Nonstop'
    df['Variant_Classification'][df['Variant_Classification'] == 'Silent'] = 'Silent'
    df['Variant_Classification'][df['Variant_Classification'] == 'Translation_Start_Site'] = 'TSS'
    #return df

def get_muta_profile(df):
    w = []
    for i in range(len(df)):
        x = df.iloc[i,:].to_numpy()
        x = " ".join(x)
        w.append(x)
    w = " ".join(w)
    w += '.'
    #print(w)
    return w


class callback(CallbackAny2Vec):
    """
    Callback to print loss after each epoch
    """
    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        if self.epoch == 0:
            print('Loss after epoch {}: {}'.format(self.epoch, loss))
        else:
            print('Loss after epoch {}: {}'.format(self.epoch, loss- self.loss_previous_step))
        self.epoch += 1
        self.loss_previous_step = loss
