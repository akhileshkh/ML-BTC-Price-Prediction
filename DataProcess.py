import numpy as np
import pandas as pd
import math

class BTC_preprocessor:
    '''
    A general utility class that provides functions for reading and
    preprocessing data for training and test on pandas dataFrame
    objects
    '''
    def read_and_preprocess(self, filename, keep_features):
        '''
        A method to read in the file and filter out the columns to be dropped

        Parameters
        ----------
        filename: str
            The name of the source file
        keep_features: str list
            A list of features to be kept in the returned DataFrame

        Returns
        ---------
        df: Pandas.DataFrame
            The loaded data with selected columns from the source file
        '''
        df = pd.read_csv(filename)
        df = df[keep_features]
        return df

    def order_of_magnitude_scale(self, df, features, features_oom):
        '''
        A method to scale features in the dataframe so that large valued
        features do not lead to gradient explosion during training

        Parameters
        ----------
        df: Pandas.Dataframe
            The DataFrame to be modified
        features: str list
            A list of features to be scaled
        features_oom: float list
            The order of magnitude by which the corresponding aforementioned
            features are to be scaled by

        Returns
        ---------
        df: Pandas.DataFrame
            The data with selected columns scaled
        '''
        identity = lambda x, y: [x, y]
        for feature, oom in map(identity, features, features_oom):
            df[feature] = df[feature]/(10.0**oom)
        return df

    def batch_normalize(self, df, features):
        '''
        A method to batch normalize features and convert them into
        samples from a distribution with 0 mean and stdev of 1

        Parameters
        ----------
        df: Pandas.Dataframe
            The DataFrame to be modified
        features: str list
            A list of features to be normalized

        Returns
        ---------
        df: Pandas.DataFrame
            The data with selected columns normalized
        '''
        for feature in features:
            mu = np.mean(df[feature])
            sigma = np.std(df[feature])
            df[feature] = (df[feature] - mu)/sigma
        return df

    def custom_transform(self, df, col_names, criterion):
        '''
        A method to transform data by applying a function to each element
        For example, absolute value, 

        Parameters
        ----------
        df: Pandas.Dataframe
            The DataFrame to be modified
        col_names: str list
            A list of columns to be transformed
        criterion:
            The function to be applied to the selected data

        Returns
        ---------
        df: Pandas.DataFrame
            The data with selected columns transformed according to
            the input criterion
        '''
        for col_name in col_names:
            df[col_name] = df[col_name].apply(criterion)
        return df

    def price_direction_label(self, df, col_names):
        '''
        A method to label the direction of data using the sign function
        Positive values are label 1 and negative with 0 for use by models
        aiming to predict the direction of the BTC price

        Parameters
        ----------
        df: Pandas.Dataframe
            The DataFrame to be modified
        col_names: str list
            A list of columns to be transformed

        Returns
        ---------
        df: Pandas.DataFrame
            The data with selected columns labeled according to their sign
        '''
        binary_label = lambda n: 1 if n > 0 else 0
        for col_name in col_names:
            df[col_name] = df[col_name].apply(binary_label)
        return df 

    def create_diffpercent_target_col(self, col_name, target_col_name, df, delta_days):
        '''
        A method to calculate the price differences between different days
        for BTC as a percentage of price (Creates new DataFrame column)

        Parameters
        ----------
        col_name: str
            Name of the column of which the difference will be calculated
        target_col_name: str
            Name of the new column to store results
        df: Pandas.DataFrame
            The DataFrame to be modified
        delta_days: int
            The number of days across which the difference is calculated

        Returns
        --------
        df: Pandas.DataFrame
            The modified DataFrame with the added difference column
        '''
        df[target_col_name] = (((df[col_name].diff())/df[col_name])*100).shift(delta_days)
        df.dropna(inplace= True)
        return df       
