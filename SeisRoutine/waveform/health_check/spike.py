import numpy as np
import SeisRoutine.utils.statistics as srus


def zscore(data, threshold=10):
    """
    Detect spikes in a 1D signal using the z-score method.

    This function identifies points in the input data that deviate significantly
    from the mean, based on a specified z-score threshold. It is useful for
    detecting outliers or transient spikes in time-series data.

    Parameters
    ----------
        data : np.ndarray
            A 1D NumPy array representing the signal (e.g., seismic trace).
        threshold : float, optional
            The z-score threshold for detecting spikes. Points with absolute
            z-scores greater than this value are considered spikes. Default is 10.

    Returns
    -------
        np.ndarray
            Indices of detected spikes in the input array.

    Examples
    --------
        >>> import numpy as np
        >>> data = np.random.normal(0, 1, 1000)
        >>> data[200] = 20  # Introduce a spike
        >>> spike_detection_zscore(data, threshold=6)
        array([200])
    """
    z_score = srus.zscore(data)
    spikes = np.where(z_score > threshold)
    return spikes[0]


def differential(data, threshold=5.0):
    """
    Detect spikes in a 1D signal based on large differences between consecutive values.

    This function identifies locations in the signal where the absolute difference 
    between adjacent samples exceeds a given threshold. Such jumps may indicate 
    sudden spikes or discontinuities in the data.

    Parameters
    ----------
        data : np.ndarray
            A 1D NumPy array representing the signal (e.g., a seismic trace).
        threshold : float, optional
            The threshold for detecting significant changes between consecutive values.
            Differences larger than this value are considered spikes. Default is 5.0.

    Returns
    -------
        np.ndarray
            Indices of the points where spikes are detected, corresponding to the 
            first sample in each spike pair (i.e., `i` such that `abs(data[i+1] - data[i]) > threshold`).

    Notes
    -----
        This method is simple and fast, but may be sensitive to noise or high-frequency content.
        It works best when the signal is relatively smooth and spikes are clearly defined by large jumps.

    Examples
    --------
        >>> import numpy as np
        >>> data = np.array([1, 1.1, 1.2, 10, 1.3])
        >>> spike_detection_diff(data, threshold=5.0)
        array([2])
    """
    diffs = np.abs(np.diff(data))
    spikes = np.where(diffs > threshold)
    return spikes[0]
