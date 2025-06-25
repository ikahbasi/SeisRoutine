import numpy as np


def zscore(data):
    """
    Compute the absolute z-score of a 1D NumPy array.

    This function calculates how far each data point in the input array deviates from
    the mean in terms of standard deviations. It returns the absolute z-scores.

    If the standard deviation is zero (i.e., the signal is flat), it returns an array
    filled with `np.inf` to indicate undefined z-scores.

    Parameters
    ----------
        data : np.ndarray
            1D array of numeric values (e.g., a seismic trace or time-series).

    Returns
    -------
        zscores : np.ndarray
            Array of absolute z-scores, same shape as `data`.

    Examples
    --------
        >>> import numpy as np
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> zscore(data)
        array([1.4142, 0.7071, 0.0, 0.7071, 1.4142])

        >>> flat = np.array([3, 3, 3])
        >>> zscore(flat)
        array([inf, inf, inf])
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:  # flat signal
        zscores = np.full_like(data, np.inf)
    else:
        zscores = np.abs((data - mean) / std)
    return zscores
