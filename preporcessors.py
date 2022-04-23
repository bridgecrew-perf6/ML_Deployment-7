
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

'''
This file include custom transfomers compatible with sciket learn transformers
'''


class TemporalVariableTransformer(BaseEstimator,TransformerMixin):
    #Temporal Elapsed time transfomer

    def __init__(self, variables, refernce_variable ):
        super().__init__()
        if not isinstance(variables,list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.reference_variable = refernce_variable
    
    def fit(self,X,y=None):
        #No need to fit, no parameters to learn from the data
     
        return self

    def transform(self,X):
        X=X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]

        return X


            # categorical missing value imputer
class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables, mappings):

        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        # we need the fit statement to accomodate the sklearn pipeline
        return self

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].map(self.mappings)

        return X



class ExtratcFirstLetter(BaseEstimator,TransformerMixin):

    def __init__(self,variables):
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')
        self.variables = variables
    
    def fit(self,X,y=None):
        return self
    
    def transform(self, X):

        # so that we do not over-write the original dataframe
        X = X.copy()
        
        for feature in self.variables:
            X[feature] = X[feature].str[0]

        return X


class Debug(BaseEstimator, TransformerMixin):

    def __init__(self,columns):
        self.columns = columns



    def transform(self, X):
        print(X.shape)
        self.shape = X.shape
        self.transformed_ds = X
       

        # what other output you want
        return X

    def fit(self, X, y=None, **fit_params):
        return self