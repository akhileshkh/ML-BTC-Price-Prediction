import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import DataProcess
import LinearRegression


# location for data source
btc_filename = 'BTC_Data_final.csv'

# features to be used in model
btc_keep_features = ['priceUSD', 'size', 'sentbyaddress', 'transactions',
                     'mining_profitability', 'sentinusd', 'marketcap', 'transactionvalue',
                     'mediantransactionvalue', 'tweets', 'google_trends', 'fee_to_reward',
                     'activeaddresses', 'top100cap', 'confirmationtime']
# features to be dropped
btc_drop_features = ['transactionfees', 'median_transaction_fee']

# order of magintude of each feature
btc_keep_features_order = [0.0, 4.0, 5.0, 5.0, 5.0,
                         1.0, 8.0, 10.0, 5.0,
                         3.0, 4.0, 1.0, 0.0,
                         5.0, 1.0, 1.0]
# number of test vs train instances
test_train_boundary = 2999

# days in the future for prediction
day_shift = 1

# name of output column
output_name = 'priceUSD'

# load dataset
preprocessor = DataProcess.BTC_preprocessor()
df = preprocessor.read_and_preprocess(btc_filename, btc_keep_features)

# create label col for train and test
df[output_name] = df[output_name].shift(0-day_shift)
df.dropna(inplace=True)

# batch normalize all kept features
df = preprocessor.batch_normalize(df, btc_keep_features)

# split dataframe into train, test
df_train = df.loc[0:test_train_boundary]
df_test = df.loc[(test_train_boundary + 1):]

X_train = np.array(df_train.drop([output_name], axis = 1))
Y_train = np.array(df_train[output_name])

X_test = np.array(df_test.drop([output_name], axis = 1))
Y_test = np.array(df_test[output_name])

# max epochs
max_iter = 1000

# learning rate
lr = 0.1

# tolerance for convergence check
tolerance = 0.0001

# penalty, l1, l2, or None
reg_type = None

# lambda for regularization
reg_rate = 0.0

# create the Linear Regression model
model = LinearRegression.LinearRegression(lr, tolerance, max_iter, reg_type, reg_rate)
model.fit(X_train, Y_train)

# obtain predictions
testout_arr = model.predict(X_test)


# data to create plot
df_display = preprocessor.read_and_preprocess('BTC_Data_final.csv', ['Date', output_name])
df_display[output_name] = df_display[output_name].shift(0-day_shift)
df_display.dropna(inplace=True)

price_col = df_display[output_name]
price_arr = np.array(price_col)

# transform output, undoing batch normalization
mu = np.mean(price_arr)
sigma = np.std(price_arr)

testout_arr = (testout_arr*sigma) + mu

# plot
plt.plot(df_display['Date'].loc[test_train_boundary + 1:], price_col.loc[test_train_boundary + 1:], label = 'Actual')
plt.plot(df_display['Date'].loc[test_train_boundary + 1:] ,testout_arr, label = 'Prediction', color = 'Orange')
plt.xlabel('Days from start')
plt.ylabel('Price')
plt.legend(['Actual','Prediction'])
plt.show()
