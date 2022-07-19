import pandas as pd


def select_patient(df, idx):
    mask = []
    for i in range(len(df)):
        if df['PATIENT_ID'][i] in idx:
            mask.append(True)
        else:
            mask.append(False)
    df = df[mask].reset_index(drop=True)
    return df
