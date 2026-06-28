import numpy as np

def safe_divide(a, b, default=0.0):
    if b == 0:
        return default
    return a / b

def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode='same')

def robust_std(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))
