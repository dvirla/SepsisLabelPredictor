from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_classif
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def feature_selector(df: pd.DataFrame, var_threshold):
    """
    Assuming df has SepsisLabel column for classification
    """
    # Variance thersholding
    sel = VarianceThreshold(var_threshold)
    sel.fit(df)
    cols_by_var = set(df.columns[sel.get_support()].tolist())

    # Regression pvals
    regression_pvals = f_classif(df.drop(columns=['SepsisLabel']), df['SepsisLabel'])[1]
    cols_from_regression = set(
        [col for (pval, col) in zip(regression_pvals.tolist(), df.columns.tolist()) if pval < 0.05])

    # Select from model
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(df.drop(columns=['SepsisLabel']), df['SepsisLabel'])
    model = SelectFromModel(lsvc, prefit=True)
    cols_from_models = set(df.drop(columns=['SepsisLabel']).columns[model.get_support()].tolist())

    # Intersecting all cols
    cols = cols_from_models.intersection(cols_by_var).intersection(cols_from_regression)

    return cols


def get_remove_cols_from_logistic_regression(X, y, coef_cutoff=0.):
    logistic_pipe = Pipeline([
        ('impute', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler()),
        ('logistic_reg', LogisticRegression(random_state=0, max_iter=100000))
    ])
    logistic_pipe.fit(X, y)
    logistic_coefs = logistic_pipe.named_steps['logistic_reg'].coef_
    below_cutoff_idxs = np.where(np.abs(logistic_coefs) <= coef_cutoff)[1]
    remove_cols = X.columns[below_cutoff_idxs]
    print("Columns removed are:", remove_cols)
    return remove_cols
