import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
import DataProcess
import LogisticRegression

# location for data source
btc_filename = 'BTC_Data_final.csv'

# features to be used in model
btc_keep_features = ['Date', 'priceUSD', 'size', 'transactions',
                     'sentinusd', 'marketcap', 'transactionvalue', 
                     'mediantransactionvalue', 'tweets', 'google_trends', 'fee_to_reward',   
                     'activeaddresses', 'top100cap', 'confirmationtime']
# features to be dropped
btc_drop_features = ['transactionfees', 'median_transaction_fee']

# number of test vs train instances
test_train_boundary = 2999

# days in the future for prediction
day_shift = 3

# name of output col
output_name = 'direction_priceUSD'

# col trasnformed to create output col
transform_col = 'priceUSD'

# load dataset
preprocessor = DataProcess.BTC_preprocessor()
df = preprocessor.read_and_preprocess(btc_filename, btc_keep_features)
df = preprocessor.create_diffpercent_target_col(transform_col, 'diffpercent', df, day_shift)

# transform percent change into direction
# adding previous 2 directions as features
df = preprocessor.price_direction_label(df, ['diffpercent'])
df['diffpercent2'] = df['diffpercent'].shift(-1)
df[output_name] = df['diffpercent2'].shift(-1)
df.dropna(inplace= True)

# store data for analysis
date_col = df['Date']
direction_col = df[output_name]
df.drop(['Date'], axis = 1, inplace = True)
btc_keep_features.remove('Date')

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
max_iter = 2000

# learning rate
lr = 0.1

#tolerance for convergence check
tolerance = 0.0001

# create the Logistic Regression model
model = LogisticRegression.WeightedLogisticRegression(lr, tolerance, max_iter)
N = X_train.shape[0]
uniform_weights = [1.0/N]*N
model.fit(X_train, Y_train, uniform_weights)

# obtain predictions
testout_arr = model.predict(X_test)

# evaluate
S = Y_test.size
correct = 0
true_pos = 0
false_pos = 0
true_neg = 0
false_neg = 0
for i in range(S):
    pred = testout_arr[i]
    test = Y_test[i]
    if pred == test:
        correct +=1
    if pred == 0 and test == 0:
        true_neg += 1
    if pred == 1 and test == 1:
        true_pos += 1
    if pred == 0 and test == 1:
        false_neg += 1
    if pred == 1 and test == 0:
        false_pos += 1

print('total: '+str(S))
print('correct: '+str(correct))
print('true up: '+str(true_pos))
print('false up: '+str(false_pos))
print('true down: '+str(true_neg))
print('false down: '+str(false_neg))
print('accuracy:' +str(correct/S))
