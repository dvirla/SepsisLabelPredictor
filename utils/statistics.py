import numpy as np
from scipy.stats import mannwhitneyu, norm
from multipy.fwer import holm_bonferroni
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def _check_continuous_feature_label_dependency_bootstrap(df, p_value_func, alpha=0.05, B=1000, seed=42):
    np.random.seed(seed)
    ignore_cols = ['SepsisLabel', 'Patient', 'Unit1', 'Unit2', 'Gender']
    res_dict = {'col': [], 'pval': [], 'With Sepsis Size': []}
    for col in df.columns:
        if col in ignore_cols:
            continue
        col_no_sepsis = df[df['SepsisLabel'] == 0][col].dropna().to_numpy()
        col_with_sepsis = df[df['SepsisLabel'] == 1][col].dropna().to_numpy()
        total_p = 0
        for b in range(B):
            col_no_sepsis_sampled = np.random.choice(col_no_sepsis, len(col_with_sepsis), replace=False)
            p = p_value_func(col_no_sepsis_sampled, col_no_sepsis)
            total_p += p
        p = total_p / B
        res_dict['col'].append(col)
        res_dict['pval'].append(p)
        res_dict['With Sepsis Size'].append(len(col_with_sepsis))
    test_df = pd.DataFrame(res_dict)
    test_df['rejected_h0_multiple_comparison'] = holm_bonferroni(test_df['pval'].values, alpha=alpha)
    return test_df.round(3).sort_values('pval')


def _check_continuous_feature_label_dependency(df, p_value_func, alpha=0.05):
    ignore_cols = ['SepsisLabel', 'Patient', 'Unit1', 'Unit2', 'Gender']
    res_dict = {'col': [], 'pval': [], 'With Sepsis Size': [], 'No Sepsis Size': []}
    for col in df.columns:
        if col in ignore_cols:
            continue
        col_no_sepsis = df[df['SepsisLabel'] == 0][col].dropna().to_numpy()
        col_with_sepsis = df[df['SepsisLabel'] == 1][col].dropna().to_numpy()
        p = p_value_func(col_with_sepsis, col_no_sepsis)
        res_dict['col'].append(col)
        res_dict['pval'].append(p)
        res_dict['With Sepsis Size'].append(len(col_with_sepsis))
        res_dict['No Sepsis Size'].append(len(col_no_sepsis))

    test_df = pd.DataFrame(res_dict)
    test_df['rejected_h0_multiple_comparison'] = holm_bonferroni(test_df['pval'].values, alpha=alpha)
    return test_df.sort_values('pval').round(3)


def mannwhitneyu_pval_func(x, y):
    _, p = mannwhitneyu(x, y)
    return p


def check_feature_label_dependency_bootstrap_mannwhitneyu(df, alpha=0.05, B=1000, seed=42):
    return _check_continuous_feature_label_dependency_bootstrap(df, mannwhitneyu_pval_func, alpha, B, seed)


def check_feature_label_dependency_mannwhitneyu(df, alpha=0.05):
    return _check_continuous_feature_label_dependency(df, mannwhitneyu_pval_func, alpha)


def wald_test(a, b):
    """This function returns the p-value and the statistic's value of wald test applied to the difference of means of a & b"""
    a_mean = a.mean()
    b_mean = b.mean()
    a_var = a.std() ** 2
    b_var = b.std() ** 2
    a_len = len(a)
    b_len = len(b)
    t_stat = (a_mean - b_mean) / ((a_var / a_len) + (b_var / b_len)) ** 0.5
    return 2*norm.cdf(-abs(t_stat))


def check_feature_label_dependency_bootstrap_wald(df, alpha=0.05, B=1000, seed=42):
    return _check_continuous_feature_label_dependency_bootstrap(df, wald_test, alpha, B, seed)


def check_feature_label_dependency_wald(df, alpha=0.05):
    return _check_continuous_feature_label_dependency(df, wald_test, alpha)


def calc_prop_test(count, nobs, value):
    stat, pval = proportions_ztest(count, nobs, value)
    return pval


def check_categorical_feature_label_dependency_bootstrap(df, alpha=0.05, B=1000, seed=42):
    np.random.seed(seed)
    z_test_res_dict = {'col': [], 'pval': [], 'With Sepsis Size': []}
    value = 0

    for col in ['Unit1', 'Unit2', 'Gender']:
        col_no_sepsis = df[df['SepsisLabel'] == 0][col].dropna().to_numpy()
        col_with_sepsis = df[df['SepsisLabel'] == 1][col].dropna().to_numpy()
        nobs = len(col_with_sepsis) * 2

        total_p = 0
        for b in range(B):
            col_no_sepsis_sampled = np.random.choice(col_no_sepsis, len(col_with_sepsis), replace=False)
            count = col_with_sepsis.sum() + col_no_sepsis_sampled.sum()
            p = calc_prop_test(count, nobs, value)
            total_p += p
        p = total_p / B

        z_test_res_dict['col'].append(col)
        z_test_res_dict['pval'].append(np.around(p, 3))
        z_test_res_dict['With Sepsis Size'].append(len(col_with_sepsis))
    test_df = pd.DataFrame(z_test_res_dict)
    test_df['rejected_h0_multiple_comparison'] = holm_bonferroni(test_df['pval'].values, alpha=alpha)
    return test_df



if __name__ == '__main__':
    df = pd.DataFrame({'SepsisLabel': [0, 0, 1, 1], 'feature': [1, 2, 3, 4]})
    check_feature_label_dependency_bootstrap_mannwhitneyu(df, alpha=0.05, B=10, seed=42)
