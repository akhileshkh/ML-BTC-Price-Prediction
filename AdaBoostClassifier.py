import random 
import numpy as np
import pandas as pd
import LogisticRegression
import math

class AdaBoostClassifier:
    '''
    A class to run the AdaBoost algorithm for use as a binary classifier
    by combining multiple weighted logistic regression models

    Attributes
    --------
    num_models: int
        The number of models to be used in this classifier
    alpha: float
        The learning rate for this model
    tolerance: float
        The minimum difference between consecutive weights to determine convergence
    max_iter: int
        The maximum number of training epochs
        
    beta_arr_: float array
        The weights given to the outputs of each model
    model_arr_:
        The models to be combined
    
    '''
    def __init__(self, num_models, alpha, tolerance, max_iter):
        self.num_models = num_models
        self.alpha = alpha
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.beta_arr_ = []
        self.model_arr_ = []

    def fit(self, X, y):
        '''
        Fits the model to the training data combining the outputs of each model
        and learning the weights to be given to the outputs of each model

        Parameters
        ----------
        X:
            The training features matrix
        y:
            The training class labels, binary 1 or 0

        '''
        N = X.shape[0]

        weights = [1.0/N]*N

        y_ = y.copy()

        # AdaBoost represents the 0 class label as -1, so transforming the labels
        for i in range(N):
            if (y_[i] == 0):
                y_[i] = -1

        for m in range(self.num_models):

            # create and fit each model
            curr_model = LogisticRegression.WeightedLogisticRegression(self.alpha, self.tolerance, self.max_iter)
            curr_model.fit(X, y, weights)
            
            # predict the labels for each model
            y_pred = curr_model.predict(X)

            # calculate epsilon, which is the weighted error
            epsilon = 0.0
            for i in range(N):
                if y_pred[i] != y[i]:
                    epsilon += weights[i]
                    
            # invert inputs and weights for models with epsilon > 0.5
            if (epsilon > 0.5):
                for i in range(N):
                    if (y_pred[i] == 1):
                        y_pred[i] == 0
                    else:
                        y_pred[i] == 1
                for i in range(curr_model.theta_.size):
                    curr_model.theta_[i] = 0.0-curr_model.theta_[i]
                    
            # add the current model
            self.model_arr_.append(curr_model)

            if epsilon == 0:
                beta = np.inf
                self.beta_arr_.append(beta)
                break
            
            # calculate and update the weights to the output of each model
            beta = 0.5*math.log((1.0 - epsilon)/epsilon)
            self.beta_arr_.append(beta)
            
            y_pred_ = y_pred.copy()
            for i in range(N):
                if (y_pred_[i] == 0):
                    y_pred_[i] = -1
                    
            # update, normalize the sample weights for the input data
            for i in range(N):
                weights[i] = weights[i]*math.exp(0.0 - beta*y_pred_[i]*y_[i])
            weights = weights/np.sum(weights)

    
    def predict(self, X):
        '''
        Predict the class labels for each data point in X using
        the leanred AdaBoost classifier and weights

        Parameters
        ----------
        X:
            The features matrix on which predictions are to be made

        Returns
        ---------
        y_pred:
            The predicted class labels
        
        '''
        N = X.shape[0]
        sum_beta_times_model = np.zeros((N,))

        # calculate the final output from the individual model outputs
        for m in range(len(self.model_arr_)):
            model = self.model_arr_[m]
            res = model.predict(X)

            for i in range(len(res)):
                if (res[i] == 0):
                    res[i] = -1

            sum_beta_times_model += self.beta_arr_[m]*res

        # convert probabilities into the binary labels    
        y_pred = np.zeros((N,))
        for i in range(N):
            if (sum_beta_times_model[i] > 0):
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred
