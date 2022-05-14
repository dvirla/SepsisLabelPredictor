import pickle
from utils import data_handler, signature, feature_selection
import csv
import sys
import pandas as pd


if __name__ == "__main__":
    data_path = sys.argv[1]
    
    X, _ = data_handler.get_model_prepared_dataset(data_path, real_test=False)
    with open('./model.pkl', 'rb') as f:
        pipe = pickle.load(f)
    
    ypreds = pipe.predict(X)
    csv_name = './prediction.csv'
    df = pd.DataFrame(list(zip(list(X.index), ypreds)), columns = ['Id', 'SepsisLabel'])
    df.to_csv(csv_name, index=False, header=False)
    