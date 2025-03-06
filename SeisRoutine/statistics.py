import numpy as np
from scipy.stats import gaussian_kde

def mode(data):
    """
    Estimate the mode of a given dataset using Kernel Density Estimation (KDE).

    Parameters:
    data (array-like): Input data for which the mode is to be estimated.

    Returns:
    float: The estimated mode of the dataset.
    """
    kde = gaussian_kde(data)
    x = np.linspace(min(data), max(data), 1000)
    mode = x[np.argmax(kde(x))]
    return mode