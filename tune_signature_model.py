import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import xgboost as xgb
from utils import data_handler, signature, feature_selection

from sklearn.model_selection import train_test_split


def train_test_split_df(df, test_size=0.3):
    X_train, X_test = train_test_split(df, test_size=test_size)
    y_train = X_train['SepsisLabel_max']
    X_train = X_train.drop(columns=['SepsisLabel_max'])
    y_test = X_test['SepsisLabel_max']
    X_test = X_test.drop(columns=['SepsisLabel_max'])
    return X_train, y_train, X_test, y_test


all_train_df = pd.read_pickle('./data/all_dfs_truncated.pkl')
all_train_df = data_handler.add_additional_features(all_train_df)

agg_dict = {col: [np.nanmean, np.nanstd, np.nanmin, np.nanmax, np.nanmedian, 'skew'] for col in all_train_df.columns}
agg_dict['SepsisLabel'] = 'max'
agg_dict['Unit1'] = np.nanmax
agg_dict['Unit2'] = np.nanmax
agg_dict['Gender'] = ['max', 'count']
agg_dict['ICULOS'] = 'max'
agg_dict['Patient'] = 'max'
agg_dict['Age'] = 'max'
agg_dict['HospAdmTime'] = 'max'

agg_df = all_train_df.groupby('Patient').agg(agg_dict)
agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]

stationary_cols = ['SepsisLabel', 'Unit1', 'Unit2', 'Gender', 'ICULOS', 'Patient', 'Age', 'HospAdmTime']

signature_cols = [col for col in all_train_df.columns if col not in stationary_cols]

args = {'gamma': 1, 'learning_rate': 0.1, 'max_depth': 9, 'subsample': 0.6}

f1_scores = {}
for truncation in tqdm([2]):
    for n_records in (40, ):
        print(truncation, n_records)

        signature_df = signature.calc_signature_for_all_df(all_train_df, signature_features=signature_cols,
                                                           truncation_level=truncation, n_records=n_records)
        print("finished calculating signature")

        signature_df[agg_df.columns] = agg_df
        df = signature_df

        X_train, y_train, X_test, y_test = train_test_split_df(df, test_size=0.3)

        remove_cols = feature_selection.get_remove_cols_from_logistic_regression(X_train, y_train, coef_cutoff=0.005)
        print("finished finding remove cols")

        pipe = Pipeline([
            ('remove_cols', data_handler.RemoveColsTransformer(remove_cols=remove_cols)),
            ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
            ('xgboost', xgb.XGBClassifier(**args))
        ])
        pipe.fit(X_train, y_train)
        print("finished fitting model")

        y_pred = pipe.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        print(f1)
        f1_scores[(truncation, n_records, 'including agg cols')] = np.round(f1, 3)

with open('param_tuning/n_score_tune_res.txt', 'a') as f:
    f.write(str(f1_scores))
