from statistics import median
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
    mode_value = x[np.argmax(kde(x))]
    return mode_value


def detect_outliers_iqr(array, multiplier=1.5):
    """
    Interquartile Range
    Tukey Fences are robust methods in detecting outliers.
    As per the Turkey method, the outliers are the points lying
    beyond the upper boundary of Q3 +1.5*IQR and the lower boundary
    of Q1 - 1.5*IQR. These boundaries are referred to as outlier fences.
    Any data beyond these fences are considered to be outliers.

    for some nonnegative constant k. John Tukey proposed this test,
    where k = 1.5 indicates an "outlier", and k = 3 indicates data
    that is "far out".
    (Fig.1 of the Kristekova_etal_GJI_2021.pdf)

    :type values: numpy.ndarray
    :param values: one-dimensional number arrays.
    :type multiplier: float
    :param multiplier: ???

    :returns:
    :type outliers: numpy.ndarray
    :param outliers: A boolean array concerning the size of input `values`.
    :type lower: float
    :param lower: the lower boundary of the fences (or Q1 - 1.5*IQR).
    :type upper: float
    :param upper: the upper boundary of the fences (or Q3 +1.5*IQR).
    """
    q1 = np.percentile(array, 25)
    q3 = np.percentile(array, 75)
    iqr_value = q3 - q1
    lower = q1 - (iqr_value * multiplier)
    upper = q3 + (iqr_value * multiplier)
    inliers_msk = (lower <= array) & (array <= upper)
    outliers_msk = ~inliers_msk
    return outliers_msk
