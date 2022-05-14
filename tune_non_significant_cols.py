import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import xgboost as xgb
from utils import data_handler, signature, feature_selection

from sklearn.model_selection import train_test_split


not_segnificant_cols = ['HospAdmTime', 'PaCO2', 'PTT', 'EtCO2', 'Platelets', 'Age','Potassium', 'Chloride', 'BaseExcess', 'Phosphate', 'SaO2']
f1_scores = {}
args = {'gamma': 0.5, 'learning_rate': 0.15, 'max_depth': 11, 'subsample': 0.6}

# for i in range(len(not_segnificant_cols)):
#     t_not_segnificant_cols = not_segnificant_cols[:-i]
for t_not_segnificant_cols in [('HospAdmTime', 'PaCO2', 'PTT', 'EtCO2', 'Platelets'), ('PaCO2', 'PTT', 'EtCO2', 'Platelets')]:
    print(t_not_segnificant_cols)
    
    X, y = data_handler.get_model_prepared_dataset('./data/train/', t_not_segnificant_cols)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    remove_cols = feature_selection.get_remove_cols_from_logistic_regression(X_train, y_train, coef_cutoff=0.005)
    print("finished finding remove cols")

    pipe = Pipeline([
        ('remove_cols', data_handler.RemoveColsTransformer(remove_cols=remove_cols)),
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('xgboost', xgb.XGBClassifier(**args))
    ])
    pipe.fit(X_train, y_train)
    print("finished fitting model for cols\n" + str(t_not_segnificant_cols))

    y_pred = pipe.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print("f1 = " + str(f1))
    f1_scores[tuple(t_not_segnificant_cols)] = np.round(f1, 3)

with open('./param_tuning/tune_non_significant_cols.py.txt', 'a') as f:
    f.write(str(f1_scores))
    
print(str(f1_scores))