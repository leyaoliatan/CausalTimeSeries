import pandas as pd
import numpy as np
from scipy.optimize import fmin_slsqp
from typing import List
from operator import add
from toolz import reduce, partial


#### Synthetic Control
def loss(W,X,Y):
    return np.sqrt(np.mean((Y - X.dot(W))**2))

def calculate_W(X,Y):
    #set initial weights as 1/number of features
    W_initial = np.ones(X.shape[1])/X.shape[1]
    #bound the weights between 0 and 1
    W = fmin_slsqp(partial(loss, X=X, Y=Y), W_initial,f_eqcons=lambda W: np.sum(W)-1, bounds=[(0,1)]*X.shape[1], disp=False)
    return W

def fit_synthetic_control(df, treated_unit):
    """Fit synthetic control model and generate predictions"""
    treated = df[df['subject_id'] == treated_unit]
    df_pre = df[df['is_post'] == False]
    df_post = df[df['is_post'] == True]
    
    # Calculate synthetic control weights using pre-treatment data
    df_pre_Y = df_pre.pivot(index='time', columns='subject_id')['Y']
    X_sc = df_pre_Y.drop(columns=treated_unit).values
    Y_sc = df_pre_Y[treated_unit].values
    weights = calculate_W(X_sc, Y_sc)
    
    # Get treated and synthetic outcomes for all periods
    #df_Y = df.pivot(index='time', columns='subject_id')['Y']
    #synthetic_outcomes = df_Y.drop(columns=treated_unit).values.dot(weights)

    # Only return the prediction on counterfactual
    df_post_Y = df_post.pivot(index='time', columns='subject_id')['Y']
    synthetic_outcomes = df_post_Y.drop(columns=treated_unit).values.dot(weights)
    
    return synthetic_outcomes