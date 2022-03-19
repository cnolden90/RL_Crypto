from data.marketdata import MarketData as md
from preprocessors.preprocessors import Preprocessors
import pandas as pd
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":
    wavelet = True
    existing_data = True
    if(existing_data):
        train_complete = pd.read_csv('train_done.csv')  
        train_complete.set_index("Unnamed: 0", inplace=True)
        validate_complete = pd.read_csv('validate_done.csv') 
        validate_complete.set_index("Unnamed: 0", inplace=True)
        test_complete = pd.read_csv('test_done.csv') 
        test_complete.set_index("Unnamed: 0", inplace=True)

    else:
        coins_to_train = ['EOS','TRX','ADA','SOL']
        ohclv_header = ['closing','low','high','open','volume']
        start_date = pd.to_datetime('2021-01-01 00:00:00')
        preprocess = Preprocessors()
        data = md.fetch_data(start_date, coins_to_train, "1h")
        data = preprocess.feature_engeneer(data, coins_to_train, start_date)
        train_data, validate_data, test_data = preprocess.split_data(data)
        normalize_columns = train_data.filter(regex='closing|low|high|open|volume').columns
        train_normalized = preprocess.normalize_data_train(train_data,ohclv_header)
        validate_normalized = preprocess.normalize_data_validate(train_data,validate_data,normalize_columns,ohclv_header, coins_to_train)
        test_normalized = preprocess.normalize_data_test(train_data,validate_data,test_data,normalize_columns,ohclv_header, coins_to_train)
        if wavelet:
            train_normalized_wavelet = preprocess.wavelet_train(train_normalized, normalize_columns)
            validate_normalized_wavelet = preprocess.wavelet_validate(train_normalized, validate_normalized, normalize_columns)
            test_normalized_wavelet = preprocess.wavelet_test(train_normalized, validate_normalized,test_normalized,normalize_columns)
            train_complete = preprocess.prepare_df(train_normalized_wavelet, coins_to_train)
            validate_complete = preprocess.prepare_df(validate_normalized_wavelet, coins_to_train)
            test_complete = preprocess.prepare_df(test_normalized_wavelet, coins_to_train)
        else:
            train_complete = preprocess.prepare_df(train_normalized, coins_to_train)
            validate_complete = preprocess.prepare_df(validate_normalized, coins_to_train)
            test_complete = preprocess.prepare_df(test_normalized, coins_to_train)
                 
        cwd = os.getcwd()
        test_done_csv = os.path.join(cwd, 'test_done.csv')
        test_complete.to_csv(test_done_csv, index=True)
        validate_done_csv = os.path.join(cwd, 'validate_done.csv')
        validate_complete.to_csv(validate_done_csv, index=True)
        train_done_csv = os.path.join(cwd, 'train_done.csv')
        train_complete.to_csv(train_done_csv, index=True)