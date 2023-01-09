import pandas as pd
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim.lr_scheduler as lrs
import matplotlib.pyplot as plt
import DataProcess

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
day_shift = 7

# load dataset
preprocessor = DataProcess.BTC_preprocessor()
df = preprocessor.read_and_preprocess(btc_filename, btc_keep_features)

# create label col for train and test
df['priceUSD'] = df['priceUSD'].shift(0-day_shift)
df.dropna(inplace=True)

# batch normalize all kept features
df = preprocessor.batch_normalize(df, btc_keep_features)

# split dataframe into train, test
df_train = df.loc[0:test_train_boundary]
df_test = df.loc[(test_train_boundary + 1):]

X_train = df_train.drop(['priceUSD'], axis = 1)
Y_train = df_train[['priceUSD']]

X_test = df_test.drop(['priceUSD'], axis = 1)
Y_test = df_test[['priceUSD']]

# convert to tensor for use by torch
X_train_tensor = torch.tensor(X_train.values.astype('float32'))
Y_train_tensor = torch.tensor(Y_train.values.astype('float32'))

X_test_tensor = torch.tensor(X_test.values.astype('float32'))
Y_test_tensor = torch.tensor(Y_test.values.astype('float32'))

# dimension for each layer
input_dim = 14

# 3 hidden layers
hidden_dim1 = 14
hidden_dim2 = 8
hidden_dim3 = 4

# output layer
output_dim = 1

# creating the model
# layer activation is by ReLU
class FeedforwardNeuralNetModel(nn.Module):

    # initializing network and layer structure
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        
        self.ReLU = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.ReLU = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)

        self.ReLU = nn.ReLU()

        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    # forward propagation
    def forward(self, x):
        
        output = self.fc1(x)

        output = self.ReLU(output)

        output = self.fc2(output)

        output = self.ReLU(output)

        output = self.fc3(output)

        output = self.ReLU(output)

        output = self.fc4(output)

        return output

# number of epochs
max_iter = 5000

# multiple attemps given nondeterministic initialization
# best perforing model will be saved
max_attempts = 40

# allowable loss
tolerance = 0.009

# variables for saving best performing model
min_loss = float('inf')
min_weights = None

# training attempts loop
for i in range(max_attempts):

    # create model
    model = FeedforwardNeuralNetModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)

    # MSE is the loss function
    criterion = nn.MSELoss()

    learning_rate = 0.25

    # Adam algorithm used for learning
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # variables to track stagnating and increasing loss to enable early exit
    stag_ct = 0
    inc_ct = 0
    stag_loss = float('inf')
    inc_loss = 0.0

    # training the model - each attempt
    for j in range(max_iter):

        
        X_train_tensor = X_train_tensor.requires_grad_()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)

        # calculating training loss
        loss = criterion(outputs, Y_train_tensor)

        # backpropagation step
        loss.backward()
        optimizer.step()

        # periodically check convergence on test set
        if j%100 == 0:
            outputs = model(X_train_tensor)
            testout = model(X_test_tensor)
            loss_test = criterion(testout, Y_test_tensor)

            # terminate this attempt in case of stagnation
            if abs(stag_loss - loss_test) < 0.0001:
                stag_ct +=1
            else:
                stag_loss = loss_test
                stag_ct = 0

            # terminate this attempt if loss increases repeatedly
            if stag_ct >= 3 or inc_ct >= 3:
                break
            
            if loss_test > inc_loss:
                inc_ct += 1
            else:
                inc_ct = 0
            inc_loss = loss_test

            # update the best model
            if(loss_test < min_loss):
                min_loss = loss_test
                min_weights = model.state_dict()
                
            # early exit if loss is acceptable
            if (loss_test < tolerance):
                break

# loading the best model from training
model = FeedforwardNeuralNetModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim)
model.load_state_dict(min_weights)

# data to create plot
df_display = preprocessor.read_and_preprocess('BTC_Data_final.csv', ['Date', 'priceUSD'])
df_display['priceUSD'] = df_display['priceUSD'].shift(0-day_shift)
df_display.dropna(inplace=True)

price_col = df_display['priceUSD']
price_arr = np.array(price_col)

# transform output, undoing batch normalization
mu = np.mean(price_arr)
sigma = np.std(price_arr)


testout = model(X_test_tensor)
loss2 = criterion(testout, Y_test_tensor)
print('test loss '+ str(loss2) )
testout_arr = np.array(testout.detach())
testout_arr = (testout_arr*sigma) + mu

# plot
plt.plot(df_display['Date'].loc[test_train_boundary + 1:], price_col.loc[test_train_boundary + 1:], label = 'Actual')
plt.plot(df_display['Date'].loc[test_train_boundary + 1:] ,testout_arr, label = 'Prediction')
plt.xlabel('Days from start')
plt.ylabel('Price')
plt.legend(['Actual','Prediction'])
plt.show()
