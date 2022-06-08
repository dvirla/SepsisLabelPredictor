import pandas as pd
from tqdm import tqdm
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pickle
from utils import signature

def load_data(data_path):
    dfs = []
    for file in tqdm(os.listdir(data_path)):
        patient_num = int(file.split('patient_')[1].split('.psv')[0])
        df = pd.read_csv(os.path.join(data_path, file), sep='|')
        # Truncate to first hour (six hours before detecting sepsis)
        if len(df[df['SepsisLabel'] == 1]) > 0:
            first_sepsis_idx = df[df['SepsisLabel'] == 1][['SepsisLabel']].idxmin().values[0]
            df = df[df.index <= first_sepsis_idx]
        df['Patient'] = patient_num
        dfs.append(df)
    all_df = pd.concat(dfs)
    all_df = all_df.reset_index(drop=True)
    return all_df


def respiration_score(fio2, pao2=92):
    if np.isnan(fio2) or fio2 == 0:
        return  0
    respiration_score = pao2 / fio2
    if respiration_score > 400:
        return 0
    elif respiration_score > 300:
        return 1
    elif respiration_score > 200:
        return 2
    elif respiration_score > 100:
        return 3
    return 4


def platelets_score(platelets):
    if np.isnan(platelets):
        return  0
    if platelets > 150:
        return 0
    elif platelets > 100:
        return 1
    elif platelets > 50:
        return 2
    elif platelets > 20:
        return 3
    return 4


def bilirubin_direct_score(bilirubin):
    if np.isnan(bilirubin):
        return  0
    if bilirubin < 1.2:
        return 0
    elif 1.2 <= bilirubin < 2:
        return 1
    elif 2 <= bilirubin < 6:
        return 2
    elif 6 <= bilirubin < 12:
        return 3
    return 4
    

def cardiovascular_hypotension_score(MAP):
    if np.isnan(MAP):
        return  0
    if MAP < 70:
        return 1
    else:
        return 0
    


def calculate_sofa(fio2, platelets, MAP, creatinine, bilirubin_direct):
    score = 0
    score += respiration_score(fio2, pao2=92)
    score += platelets_score(platelets)
    score += bilirubin_direct_score(bilirubin_direct)
    return score


def add_additional_features(df):
    # Adding columns
    sofa_features = ['FiO2', 'Platelets', 'MAP', 'Creatinine', 'Bilirubin_total']
    df['SOFA'] = df[sofa_features].apply(
        lambda x:  calculate_sofa(x.FiO2, x.Platelets, x.MAP, x.Creatinine, x.Bilirubin_total), axis=1)
    df['HR/SBP'] = df['HR']/df['SBP']
#     df['BUN/CRT'] = df['BUN']/df['Creatinine']
    return df


def leave_only_reasonable_values(all_df):
    with open('data/cutoff_dicts/low_cutoff.pkl', 'rb') as f:
        low_cutoff_dict = pickle.load(f)
    with open('data/cutoff_dicts/high_cutoff.pkl', 'rb') as f:
        high_cutoff_dict = pickle.load(f)
    for col in low_cutoff_dict.keys():
        all_df.loc[(all_df[col] < low_cutoff_dict[col]) | (all_df[col] > high_cutoff_dict[col]), col] = np.nan
    return all_df


class RemoveColsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_cols=tuple()):
        self.remove_cols = remove_cols

    def fit(self, X, y):
        return self

    def transform(self, x):
        keep_cols = [col for col in x.columns if col not in self.remove_cols]
        return x.loc[:, keep_cols]
    
class CustomImputerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mean_series = None

    def fit(self, X, y):
        self.mean_series = X.mean()
        self.mean_series['Unit1_nanmax'] = -1
        self.mean_series['Unit2_nanmax'] = -1
        return self

    def transform(self, x):
        return x.fillna(self.mean_series)
        
    
def get_model_prepared_dataset(data_folder, not_segnificant_cols=None):
    all_train_df = load_data(data_folder)
    all_train_df = leave_only_reasonable_values(all_train_df)
    all_train_df = add_additional_features(all_train_df)
    
    n_records = 40
    truncation = 2
    nan_cols = ['Lactate', 'Alkalinephos', 'AST', 'TroponinI', 'Fibrinogen', 'Bilirubin_direct']
    if not_segnificant_cols is None:
        not_segnificant_cols = ['PaCO2', 'PTT', 'EtCO2', 'Platelets']
    d_cols = set.union(set(nan_cols), set(not_segnificant_cols))
    agg_dict = {col: [np.nanmean, np.nanstd, np.nanmin, np.nanmax, np.nanmedian, 'skew'] for col in all_train_df.columns if col not in d_cols and col not in ['Patient', 'SepsisLabel']}
    agg_dict['SepsisLabel'] = 'max'
    agg_dict['Unit1'] = np.nanmax
    agg_dict['Unit2'] = np.nanmax
    agg_dict['Gender'] = [np.nanmax, 'count']
    agg_dict['ICULOS'] = 'max'
    agg_dict['Age'] = 'max'
    agg_dict['HospAdmTime'] = 'max'

    stationary_cols = ['SepsisLabel', 'Unit1', 'Unit2', 'Gender', 'ICULOS', 'Patient', 'Age', 'HospAdmTime']
    signature_cols = [col for col in all_train_df.columns if col not in stationary_cols and col not in d_cols]

    signature_df = signature.calc_signature_for_all_df(all_train_df, signature_features=signature_cols, 
                                                       truncation_level=truncation, n_records=n_records)
    signature_df = signature_df.set_index('Patient')
    
    n_records_for_patient = 5
    all_train_df = all_train_df.groupby('Patient').tail(n_records_for_patient).reset_index(drop=True).sort_values(['Patient', 'ICULOS'])
    df = all_train_df.groupby('Patient').agg(agg_dict)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    signature_df[df.columns] = df
    df = signature_df    
    nan_skew_cols = ['Calcium_skew', 'Creatinine_skew', 'Phosphate_skew', 'Bilirubin_total_skew']
    X = df.drop(columns=['SepsisLabel_max'] + nan_skew_cols)
    y = df['SepsisLabel_max']
    
    return X, y