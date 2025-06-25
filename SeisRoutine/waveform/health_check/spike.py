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
