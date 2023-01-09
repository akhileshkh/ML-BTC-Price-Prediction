import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import DataProcess
import LinearRegression

# location for data source
btc_filename = 'BTC_Data_final.csv'

# features to be used in model
btc_keep_features = ['Date', 'priceUSD', 'size', 'sentbyaddress', 'transactions',
                     'mining_profitability', 'sentinusd', 'marketcap', 'transactionvalue',
                     'mediantransactionvalue', 'tweets', 'google_trends', 'fee_to_reward',
                     'activeaddresses', 'top100cap', 'confirmationtime']
# features to be dropped
btc_drop_features = ['transactionfees', 'median_transaction_fee']

# number of test vs train instances
test_train_boundary = 2999

# days in the future for prediction
day_shift = 1

# name of output col
output_name = 'delta_priceUSD'

# col trasnformed to create output col
transform_col = 'priceUSD'

# load dataset
preprocessor = DataProcess.BTC_preprocessor()
df = preprocessor.read_and_preprocess(btc_filename, btc_keep_features)
df = preprocessor.create_diffpercent_target_col(transform_col, output_name, df, day_shift)

# transform percent change into change magnitude with absolute value
df = preprocessor.custom_transform(df, [output_name], abs)

# store data for display
date_col = df['Date']
delta_col = df[output_name]
df.drop(['Date'], axis = 1, inplace = True)
btc_keep_features += [output_name]
btc_keep_features.remove('Date')

# find mean and stdev of output col
mu = np.mean(df[output_name])
sigma = np.std(df[output_name])

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
testout_arr = ((testout_arr*sigma) + mu)

# plot
plt.plot(date_col.loc[test_train_boundary + 1:], delta_col.loc[test_train_boundary + 1:])
plt.plot(date_col.loc[test_train_boundary + 1:], testout_arr)

plt.xlabel('Days from start')
plt.ylabel('Price Change Magnitude')
plt.legend(['Actual','Prediction'])
plt.show()
