# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 00:22:58 2019

@author: Hard Med lenovo
"""
from keras import backend as K
import numpy as np


def mean_square_error(y, y_pred):
    if y_pred.ndim==2:
        y_pred = y_pred[:,0]
    return np.mean((y-y_pred)**2)


def mse_asymetric_lower(y_true, y_pred):
    """
    mean square error which give a different weights if the error is
    below or above the real value
    the weights i 1.5 if the predicted value is below and 1 otherwise
    """
    diff = K.square(y_pred - y_true)
    loss = 1.5 * diff * K.cast((y_pred < y_true), K.floatx()) + diff * K.cast((y_pred >= y_true), K.floatx())
    return K.mean(loss, axis=-1)


def mse_asymetric_higher(y_true, y_pred):
    """
    mean square error which give a different weights if the error is
    below or above the real value
    the weights i 1.5 if the predicted value is above and 1 otherwise
    """
    diff = K.square(y_pred - y_true)
    loss = diff * K.cast((y_pred < y_true), K.floatx()) + 1.5 * diff * K.cast((y_pred >= y_true), K.floatx())
    return K.mean(loss, axis=-1)
