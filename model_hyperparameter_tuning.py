import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from utils import data_handler
import multiprocessing as mp
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from utils import feature_selection
import random

if __name__ == "__main__":
    random.seed(13)
    np.random.seed(13)
    with open('data/X_y.pkl', 'rb') as f:
        X, y = pickle.load(f)
    remove_cols = feature_selection.get_remove_cols_from_logistic_regression(X, y, coef_cutoff=0.005)

    pipe = Pipeline([
        ('remove_cols', data_handler.RemoveColsTransformer(remove_cols=remove_cols)),
    #     ('impute', data_handler.CustomImputerTransformer(is_svm=True)),
        ('impute', data_handler.CustomImputerTransformer()),
    #     ('scaler', StandardScaler()),
        ('RF', RandomForestClassifier(random_state=42))
#         ('xgboost', xgb.XGBClassifier())
    ])
    param_grid = { 
    'RF__n_estimators': [100, 200, 500],
    'RF__max_features': ['auto', 'sqrt', 'log2'],
    'RF__max_depth' : [4,5,6,7,8, None],
    'RF__criterion' :['gini', 'entropy', 'log_loss']
    }
    print('Starting grid search')
    search = GridSearchCV(pipe, param_grid, n_jobs=mp.cpu_count(), scoring='f1', verbose=2)
    search.fit(X, y)
    
    print("Best parameter (CV score=%0.3f):" % search.best_score_)
    print(search.best_params_)
    
#     with open('./search_best_params.pkl', 'wb') as f:
#         pickle.dump(search.best_params_, f)
    with open('param_tuning/grid_serach_object_rf.pkl', 'wb') as f:
        pickle.dump(search, f)
    print(search.best_params_)