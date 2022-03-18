from data.marketdata import MarketData as md
from preprocessors.preprocessors import Preprocessors as pp


if __name__ == "__main__":
    wavelet = True
    coins_to_train = ['EOS','TRX','ADA','SOL']
    data = md.fetch_data(coins_to_train, "1h")
    ohclv_header = ['closing','low','high','open','volume']
    train_data, validate_data, test_data = pp.split_data(data)
    normalize_columns = train_data.filter(regex='closing|low|high|open|volume').columns
    train_normalized = pp.normalize_data_train(train_data,ohclv_header)
    # ToDo: mistake in taking the normalized train data 
    validate_normalized = pp.normalize_data_validate(train_data,validate_data,normalize_columns,ohclv_header, coins_to_train)
    # ToDo: mistake in taking the normalized train data 
    test_normalized = pp.normalize_data_test(train_data,validate_data,test_data,normalize_columns,ohclv_header, coins_to_train)
    if wavelet:
        train_normalized_wavelet = pp.wavelet_train(train_normalized, normalize_columns)
        validate_normalized_wavelet = pp.wavelet_validate(train_normalized, validate_normalized, normalize_columns)
        test_normalized_wavelet = pp.wavelet_test(train_normalized, validate_normalized,test_normalized,normalize_columns)
        train_complete = pp.prepare_train_df(train_normalized_wavelet, coins_to_train)
        validate_complete = pp.prepare_validate_df(validate_normalized_wavelet, coins_to_train)
        test_complete = pp.prepare_test_df(test_normalized_wavelet, coins_to_train)
        
    train_complete = pp.prepare_train_df(train_normalized, coins_to_train)
    validate_complete = pp.prepare_validate_df(validate_normalized, coins_to_train)
    test_complete = pp.prepare_test_df(test_normalized, coins_to_train)