
from random import seed
import numpy as np
import pandas as pd


def create_numpy_ones_array(shape):
    array=None
	#TODO
    array = np.ones(shape)
    return array

def create_numpy_zeros_array(shape):
    array=None
    #TODO
    array = np.zeros(shape)
    return array
  
def create_identity_numpy_array(order):
    array=None
    #TODO
    array=np.identity(order)
    return array

def matrix_cofactor(array):
    determinant = np.linalg.det(array)
    array1=None
    #TODO
    array1 = np.linalg.inv(array).T * determinant
    return array1

def f1(X1,coef1,X2,coef2,seed1,seed2,seed3,shape1,shape2):
    ans = None
    W1 = None
    W2 = None
    np.random.seed(seed1)
    W1 = np.random.random(shape1)
    np.random.seed(seed2)
    W2 = np.random.random(shape2)
    f1 = W1.shape
    f2 = W2.shape
    f3 = X1.shape
    f4 = X2.shape
    if(f1[1] != f3[0] or f2[1] != f4[0]):
        ans = -1
        return ans
    X1=X1**coef1
    X2=X2**coef2
    W1 = W1.dot(X1)
    W2 = W2.dot(X2)
    if(W1.shape == W2.shape):
        ans = W1+W2+seed3
        return ans
    else:
        ans = -1
        return ans

        # TODO


def fill_with_mode(filename, column):
    df= pd.read_csv(filename)
    df[column]=df[column].fillna(df[column].mode()[0])
    return df

def fill_with_group_average(df, group, column):
    df[column]=df[column].fillna(df.groupby(group)[column].transform('mean'))
    return df


def get_rows_greater_than_avg(df, column):
    # df=None
    # return df
    return df[df[column] > df[column].mean()]

