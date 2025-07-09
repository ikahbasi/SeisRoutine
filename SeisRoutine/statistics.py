from statistics import median
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import skew, zscore


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
    q1, q3 = np.percentile(array, [25, 75])
    iqr_value = q3 - q1
    lower = q1 - (iqr_value * multiplier)
    upper = q3 + (iqr_value * multiplier)
    inliers_msk = (lower <= array) & (array <= upper)
    outliers_msk = ~inliers_msk
    return outliers_msk


def detect_outliers_ztest(array, threshold=3):
    """
    Detect and remove outliers using Z-score analysis.

    Identifies outliers as values where the absolute Z-score exceeds
    the threshold, then returns a filtered array with outliers removed. Uses the
    skewness of the data for logging purposes (though not for outlier
    calculation).

    Parameters
    ----------
        array : np.ndarray
            1D array of numerical values to analyze.
        threshold : float, optional (default=3)
            Z-score threshold for outlier detection. Higher values make the test
            more conservative (fewer outliers detected). Common values:
            - 2.58 (~99% confidence for normal data)
            - 3.0 (~99.7% confidence for normal data)

    Returns
    -------
        np.ndarray
            Copy of the input array with outliers removed.

    Notes
    -----
        - Z-scores are calculated as (x - mean) / std.
        - Assumes approximately normal distribution for accurate results.
        - Logs the count of detected outliers at INFO level.

    Examples
    --------
        >>> data = np.array([1, 2, 3, 100])
        >>> ztest(data, threshold=2)
        array([False, False, False, True])
    """
    # data_skewness = skew(array)
    z_scores = zscore(array)
    outliers_msk = np.abs(z_scores) > threshold
    return outliers_msk


def compute_distance_weights(array, epsilon=1e-10):
    """
    Compute distance weights for Distance-Weighted Averaging (DWA) as per
    Eq.4 in:
        https://doi.org/10.1093/gji/ggae049

    The weights are calculated as 1/sum(|x_i - x_j|) for each element x_i.

    Parameters
    ----------
        array : np.ndarray
            Input array of values (1D or 2D). If 1D, will be converted to 2D row
            vector.

    Returns
    -------
        np.ndarray
            Array of weights with same number of elements as input array.

    Notes
    -----
        - For numerical stability, adds small epsilon to denominator
        - Handles both vector and matrix inputs automatically
    """
    array = np.atleast_2d(array)  # Ensure 2D without copying if already 2D
    if array.shape[0] > 1:
        raise ValueError("Input must be a single vector (shape: [1, N] or [N])")
    
    diffs = np.abs(array - array.T)  # Pairwise differences
    weights = 1 / (diffs.sum(axis=1) + epsilon)
    weights = weights.squeeze()  # Return as 1D if input was 1D
    return weights


def distance_weighted_average(array):
    """
    Compute Distance-Weighted Average (DWA) as per Eq.3 in:
    https://doi.org/10.1093/gji/ggae049

    Parameters
    ----------
        array : np.ndarray
            Input array of values (1D or 2D)

    Returns
    -------
        float or np.ndarray
            The weighted average value(s)

    Examples
    --------
        >>> distance_weighted_average(np.array([1, 2, 3]))
        2.0
    """
    weights = compute_distance_weights(array)
    average = np.average(array, weights=weights)
    return average
