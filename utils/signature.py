import numpy as np
import pandas as pd
import esig.tosig as ts

TRUNCATION_LEVEL = 2
N_RECORDS = 10

def calc_signature_for_all_df(all_df, signature_features, truncation_level=TRUNCATION_LEVEL, n_records=N_RECORDS):
    """
        all_df: df with cols ['ICULOS', 'Patient'] + [features for signature]
        signature_features: the features for which to calculate the signature
        truncation_level: depth parameter for signature calculations

        The all_df needs to be without fillna. nans will be handeled in the folowing way:
        - a forward fill will be applied to each feature
        - nans before the first featur's value will be discarded
        - a vector indicating if the value was nan will be added to the signature calculation
        - if the entire feature vector was nan returns nan
        returns:
        New dataframe with a signature representaion for every feature in [features for signature]

    """
    signature_dict_list = []
    for patient, patient_df in all_df.groupby('Patient'):
        signature_dict = calc_signature_for_patient_df(patient_df, signature_features, truncation_level, n_records)
        signature_dict['Patient'] = patient
        signature_dict_list.append(signature_dict)
    return pd.DataFrame(signature_dict_list)


def get_expected_shape_of_signature(data_dim_1, truncation_level=TRUNCATION_LEVEL):
    return len(calc_signature_from_stream(np.random.random(size=(2, data_dim_1)), truncation_level))


def calc_signature_for_patient_df(df, signature_features, truncation_level=TRUNCATION_LEVEL, n_records=N_RECORDS):
    time_vec = df['ICULOS'].values[:-1]
    signature_col_names = []
    signature_vecs = []
    for feature in signature_features:
        nan_vec = df[feature].isna().values.astype(int)[:-1]
        feature_vec = df[feature].fillna(method="ffill").values
        feature_vec[~np.isnan(feature_vec)] = np.cumsum(feature_vec[~np.isnan(feature_vec)])
        lead = feature_vec[1:]
        lag = feature_vec[:-1]
        stream = np.column_stack([time_vec, lead, lag, nan_vec]).astype(float)
        stream = stream[~np.isnan(stream).any(axis=1)]
        stream_shape = stream.shape
        stream = stream[-n_records:, :]
        if stream.size == 0:
            signature = np.array([np.nan] * get_expected_shape_of_signature(stream_shape[1], truncation_level))
        else:
            signature = calc_signature_from_stream(stream, truncation_level)

        signature_vecs.append(signature)
        signature_col_names += [feature + '_' + str(i) for i in range(len(signature))]
    patient_signature_vec = np.concatenate(signature_vecs)
    signature_dict = {col_name: signature_value for col_name, signature_value in
                      zip(signature_col_names, patient_signature_vec)}
    return signature_dict


def calc_signature_from_stream(stream, truncation_level=TRUNCATION_LEVEL):
    return ts.stream2logsig(stream, truncation_level)


if __name__ == '__main__':
    df = pd.DataFrame({'Patient': [1, 1, 1, 1, 2, 2, 2, 2],
                       'ICULOS': [1, 2, 3, 4, 1, 2, 3, 4],
                       'feature1': [1, 3, 8, 3, None, 2, 3, None],
                       'feature2': [4, 8, 6, -1, None, None, None, None]})
    signature_df = calc_signature_for_all_df(df, ['feature1', 'feature2'], truncation_level=TRUNCATION_LEVEL, n_records=5)
    a = 2
