# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 21:36:13 2022

@author: christian.nolden
"""
import pandas as pd
import numpy as np
import ta
from statistics import mean
import pywt
from statsmodels.robust import mad


class Preprocessors:
    
    def preprocess_cylicial_features(data, coin):
        """ cylicial feature encoding following: https://towardsdatascience.com/cyclical-features-encoding-its-about-time-ce23581845ca"""  
        data['dayOfWeek'] = data['%s:date' %coin].dt.dayofweek
        dataDayOfWeekMax = data['dayOfWeek'].max()
        data['%s:dayOfWeek_sin' % coin] = np.sin(2 * np.pi * data['dayOfWeek']/dataDayOfWeekMax)
        data['%s:dayOfWeek_cos' % coin] = np.cos(2 * np.pi * data['dayOfWeek']/dataDayOfWeekMax)
        """ Date cos & sin """
        data['dayInYear'] = data['%s:date' % coin].apply(lambda x: int(x.strftime('%j')))
        dataDayInYearMax = data['dayInYear'].max()
        data['%s:dayInYear_sin' % coin] = np.sin(2 * np.pi * data['dayInYear']/dataDayInYearMax)
        data['%s:dayInYear_cos' % coin] = np.cos(2 * np.pi * data['dayInYear']/dataDayInYearMax)
        """ Time cos & sin """
        data["timeOfDay"] = (data['%s:date' % coin].dt.hour*60) + data['%s:date' % coin].dt.minute
        dataTimeOfDayMax = data['timeOfDay'].max()
        data['%s:timeOfDay_sin' % coin] = np.sin(2 * np.pi * data['timeOfDay']/dataTimeOfDayMax)
        data['%s:timeOfDay_cos' % coin] = np.cos(2 * np.pi * data['timeOfDay']/dataTimeOfDayMax)
        """ Month cos & sin """
        data["monthInYear"] = data['%s:date' % coin].dt.month
        dataMonthInYearMax = data['monthInYear'].max()
        data['%s:monthInYear_sin' % coin] = np.sin(2 * np.pi * data['monthInYear']/dataMonthInYearMax)
        data['%s:monthInYear_cos'% coin] = np.cos(2 * np.pi * data['monthInYear']/dataMonthInYearMax)
       
        data = data.set_index('%s:date' % coin)
        
        #normalize the data over MA
        data['%s:close_ma_20' % coin] = ta.trend.ema_indicator(data['%s:close' % coin], window=20, fillna=True)
        data['%s:closing_smooth' % coin] = (data['%s:close' % coin] - data['%s:close_ma_20' % coin]) / data['%s:close_ma_20' % coin] 
        data['%s:open_smooth' % coin] = (data['%s:open' % coin] - data['%s:close_ma_20' % coin]) / data['%s:close_ma_20' % coin]
        data['%s:high_smooth' % coin] = (data['%s:high' % coin] - data['%s:close_ma_20' % coin]) / data['%s:close_ma_20' % coin]
        data['%s:low_smooth' % coin] = (data['%s:low' % coin] -data['%s:close_ma_20' % coin]) / data['%s:close_ma_20' % coin]
        data['%s:volume_ma_20' % coin] = ta.trend.ema_indicator(data['%s:volume' % coin], window=20, fillna=True)
        data['%s:volume_smooth' % coin] = (data['%s:volume' % coin] - data['%s:volume_ma_20' % coin]) / data['%s:volume_ma_20' % coin]
       
        columns_to_drop = ['timeOfDay', 'dayOfWeek',  'monthInYear', 'dayInYear', \
                           ('%s:tradecount' % coin),('%s:unix' % coin), \
                           ('%s:open' % coin), ('%s:close' % coin), ('%s:high' % coin), ('%s:low' % coin), ('%s:volume' % coin), \
                          ('%s:close_ma_20' % coin), ('%s:volume_ma_20' % coin)]
        data = data.drop(columns_to_drop, axis=1)
        return data
    
    def split_data(data):
        num_datapoints = data.shape[0]
        step_size = int(0.1 * num_datapoints)
        train_data = data[:-2*step_size]
        validate_data = data[-2*step_size:-step_size]
        test_data = data[-step_size:]
        return train_data, validate_data, test_data
    
    def normalize_data_train(train_data, ohclv_header):
        train_normalized = train_data.copy()
        # Calculate mean across coins per OHCLV data
        for header in ohclv_header:
            header_to_normalize = train_data.filter(regex=header).columns
            train_normalized_temp = train_normalized[header_to_normalize]
            train_list = pd.concat((train_normalized_temp.iloc[:,0] , train_normalized_temp.iloc[:,1] , train_normalized_temp.iloc[:,2] , train_normalized_temp.iloc[:,3]), axis=0)
            mean = np.mean(train_list.dropna())
            std = np.std(train_list.dropna())
            for i in header_to_normalize:
                train_normalized[i] = (train_normalized[i] - mean)  / (3 * std)
        return train_normalized
        
    def normalize_data_validate(train_data, validate_data, normalize_columns, ohclv_header, coins_to_train):
        train = train_data.copy()
        validate_normalized = pd.DataFrame()
        for header in ohclv_header:
            header_to_normalize = train.filter(regex=header).columns
            temp = train[header_to_normalize]
            validate_normalized_temp = validate_data[header_to_normalize]
            for j in range(validate_data.shape[0]):
                list_datapoints = []
                temp = np.append(temp, np.expand_dims(validate_normalized_temp.iloc[j,:], axis=0), axis=0)
                for coin in range(len(coins_to_train)):
                    list_datapoints.extend(temp[:,coin])
                for i in range(validate_normalized_temp.shape[1]):
                    index_row = temp.shape[0] - 1
                    mean = np.nanmean(list_datapoints)
                    std = np.nanstd(list_datapoints)
                    validate_normalized.loc[j,header_to_normalize[i]] = (temp[index_row,i] - mean)  / (3 * std)
            
        validate_data.reset_index(inplace=True)
        validate_data = validate_data.drop(validate_data.filter(regex='closing|low|high|open|volume').columns, axis=1)
        validate_normalized = pd.concat((validate_data,validate_normalized),axis=1)
        validate_normalized.set_index('index')
        return validate_normalized
    
    def normalize_data_test(train_data, validate_data, test_data, normalize_columns, ohclv_header, coins_to_train):
        temp_train = train_data.copy().filter(regex='closing|low|high|open|volume')
        temp_val = validate_data.copy().filter(regex='closing|low|high|open|volume')
        train_val = pd.concat((temp_train, temp_val),axis=0)
        test_normalized = pd.DataFrame()
        for header in ohclv_header:
            header_to_normalize = temp_train.filter(regex=header).columns
            temp = train_val[header_to_normalize]
            test_normalized_temp = test_data[header_to_normalize]
            for j in range(test_data.shape[0]):
                list_datapoints = []
                temp = np.append(temp, np.expand_dims(test_normalized_temp.iloc[j,:], axis=0), axis=0)
                for coin in range(len(coins_to_train)):
                    list_datapoints.extend(temp[:,coin])
                for i in range(test_normalized_temp.shape[1]):
                    index_row = temp.shape[0] - 1
                    mean = np.nanmean(list_datapoints)
                    std = np.nanstd(list_datapoints)
                    test_normalized.loc[j,header_to_normalize[i]] = (temp[index_row,i] - mean)  / (3 * std)
        test_data.reset_index(inplace=True)
        test_data = test_data.drop(normalize_columns, axis=1)
        test_normalized = pd.concat((test_data,test_normalized),axis=1)
        test_normalized.set_index('index')
        return test_normalized
                

    def waveletSmooth(self, x, wavelet="haar", level=2, declevel=2):
        # calculate the wavelet coefficients
        coeff = pywt.wavedec( x, wavelet, mode='periodization',level=declevel,axis=0 )
        # calculate a threshold
        sigma = mad(coeff[-level])
        #print("sigma: ",sigma)
        uthresh = sigma * np.sqrt( 2*np.log( len( x ) ) )
        #print("uthresh: ", uthresh)
        coeff[1:] = ( pywt.threshold( i, value=uthresh, mode="hard" ) for i in coeff[1:] )
        # reconstruct the signal using the thresholded coefficients
        y = pywt.waverec( coeff, wavelet, mode='periodization',axis=0 )
        return y
    
    
    def wavelet_train(self, train_data, normalize_columns):
        train_normalized = train_data.copy()
        for i in normalize_columns:
            train_normalized[i] = self.waveletSmooth(train_normalized[i], level=1)[-len(train_data):]               
        return train_normalized
    
    def wavelet_validate(self, train_data, validate_data, normalize_columns):
        temp = train_data.copy().filter(regex='closing|low|high|open|volume')
        feats_norm_validate_WT = validate_data.copy().filter(regex='closing|low|high|open|volume')
        for j in range(validate_data.shape[0]):
            temp = np.append(temp, np.expand_dims(feats_norm_validate_WT.iloc[j,:], axis=0), axis=0)
            for i in range(feats_norm_validate_WT.shape[1]):
                feats_norm_validate_WT.iloc[j,i] = self.waveletSmooth(temp[:,i], level=1)[-1]
        validate_data.reset_index(inplace=True)
        validate_data = validate_data.drop(normalize_columns, axis=1)
        validate_normalized_wavelet = pd.concat((validate_data,feats_norm_validate_WT),axis=1)
        validate_normalized_wavelet = validate_normalized_wavelet.drop('level_0', axis=1)
        validate_normalized_wavelet = validate_normalized_wavelet.set_index('index')    
        return validate_normalized_wavelet
    
    def wavelet_test(self, train_data, validate_data, test_data, normalize_columns):
        temp_train = train_data.copy().filter(regex='closing|low|high|open|volume')
        temp_val = validate_data.copy().filter(regex='closing|low|high|open|volume')
        temp = pd.concat((temp_train, temp_val),axis=0)
        feats_norm_test_WT = test_data.copy().filter(regex='closing|low|high|open|volume')
        for j in range(test_data.shape[0]):
           #first concatenate train with the latest validation sample
            temp = np.append(temp, np.expand_dims(feats_norm_test_WT.iloc[j,:], axis=0), axis=0)
            for i in range(feats_norm_test_WT.shape[1]):
                feats_norm_test_WT.iloc[j,i] = self.waveletSmooth(temp[:,i], level=1)[-1]
        test_data.reset_index(inplace=True)
        test_data = test_data.drop(normalize_columns, axis=1)
        test_normalized_wavelet = pd.concat((test_data,feats_norm_test_WT),axis=1)
        test_normalized_wavelet = test_normalized_wavelet = test_normalized_wavelet.drop('level_0', axis=1)
        test_normalized_wavelet = test_normalized_wavelet.set_index('index')
                
    def prepare_train_df(train_data, coins_to_train):
        data = pd.DataFrame()
        for coin in coins_to_train:
           headers = train_data.filter(regex=coin).columns
           temp = train_data[headers]
           temp.columns = temp.columns.str[4:]
           temp['ticker'] = coin
           temp = temp.rename(columns={'closing_smooth': 'close', 
                                      'open_smooth': 'open',
                                      'high_smooth': 'high',
                                      'low_smooth': 'low',
                                      'volume_smooth': 'volume'})
           data = pd.concat((data,temp), axis=0)
           data.dropna(how="any",inplace=True)       
        training_data_final = data.sort_index()
        temp_encoder = pd.get_dummies(training_data_final['ticker'])
        training_data_final = pd.concat((training_data_final,temp_encoder), axis=1)
        training_data_final = training_data_final.reset_index()
        training_data_final.index = training_data_final['index'].factorize()[0]
        training_data_final = training_data_final.drop(['index', 'ticker'], axis=1)
                
    def prepare_validate_df(validate_data, coins_to_train):
        data = pd.DataFrame()
        for coin in coins_to_train:
            headers = validate_data.filter(regex=coin).columns
            temp = validate_data[headers]
            temp.columns = temp.columns.str[4:]
            temp['ticker'] = coin
            temp = temp.rename(columns={'closing_smooth': 'close', 
                                      'open_smooth': 'open',
                                      'high_smooth': 'high',
                                      'low_smooth': 'low',
                                      'volume_smooth': 'volume'})
            data = pd.concat((data,temp), axis=0)
            data.dropna(how="any",inplace=True)       
        training_data_final = data.sort_index()
        temp_encoder = pd.get_dummies(training_data_final['ticker'])
        training_data_final = pd.concat((training_data_final,temp_encoder), axis=1)
        training_data_final = training_data_final.reset_index()
        training_data_final.index = training_data_final['index'].factorize()[0]
        training_data_final = training_data_final.drop(['index', 'ticker'], axis=1)
        
    def prepare_test_df(test_data, coins_to_train):
        data = pd.DataFrame()
        for coin in coins_to_train:
            headers = test_data.filter(regex=coin).columns
            temp = test_data[headers]
            temp.columns = temp.columns.str[4:]
            temp['ticker'] = coin
            temp = temp.rename(columns={'closing_smooth': 'close', 
                                      'open_smooth': 'open',
                                      'high_smooth': 'high',
                                      'low_smooth': 'low',
                                      'volume_smooth': 'volume'})
            data = pd.concat((data,temp), axis=0)
            data.dropna(how="any",inplace=True)       
        training_data_final = data.sort_index()
        temp_encoder = pd.get_dummies(training_data_final['ticker'])
        training_data_final = pd.concat((training_data_final,temp_encoder), axis=1)
        training_data_final = training_data_final.reset_index()
        training_data_final.index = training_data_final['index'].factorize()[0]
        training_data_final = training_data_final.drop(['index', 'ticker'], axis=1)