import numpy as np
from scipy.signal import find_peaks
import pywt
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


def mad(data, threshold=6):
    """
    Detect spikes in a 1D signal using the Median Absolute Deviation (MAD) method.

    This function identifies spikes by computing a modified z-score based on the 
    Median Absolute Deviation (MAD), which is a robust measure of variability. 
    It is particularly effective for detecting outliers in signals that may contain 
    extreme values or are not normally distributed.

    Parameters
    ----------
        data : np.ndarray
            A 1D NumPy array representing the signal (e.g., a seismic trace).
        threshold : float, optional
            The threshold for detecting spikes. Data points with a modified z-score 
            greater than this value are considered spikes. Default is 6.

    Returns
    -------
        np.ndarray
            Indices of the detected spikes in the input data.

    Notes
    -----
        - This method is robust to outliers and works well when the data may have 
        heavy-tailed distributions.

    Examples
    --------
        >>> import numpy as np
        >>> data = np.random.normal(0, 1, 1000)
        >>> data[300] = 15  # Introduce a spike
        >>> spike_detection_mad(data, threshold=6)
        array([300])
    """
    # normalizing MAD to be comparable to the standard deviation under
    # the assumption of a normal (Gaussian) distribution.
    normalizing_factor = 0.6745
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_zscore = np.abs(normalizing_factor * (data - median) / (mad + 1e-8))
    return np.where(modified_zscore > threshold)[0]


def prominence(data, prominence=5):
    """
    Detect spikes in a 1D signal using peak prominence.

    This function uses the `scipy.signal.find_peaks` method to identify spikes 
    based on the prominence of peaks in the absolute value of the signal. 
    Prominence measures how much a peak stands out relative to its surrounding values, 
    making this method effective for detecting sharp, isolated spikes.

    Parameters
    ----------
        trace : np.ndarray
            A 1D NumPy array representing the input signal (e.g., seismic waveform).
        prominence : float, optional
            Minimum required prominence of peaks to be considered as spikes. 
            Larger values detect only more significant spikes. Default is 5.

    Returns
    -------
        np.ndarray
            Indices of the detected spikes (peaks) in the signal.

    Notes
    -----
        - The function operates on the absolute value of the input signal to detect 
        both positive and negative spikes.
        - Prominence helps filter out low-amplitude fluctuations or noise.

    Examples
    --------
        >>> import numpy as np
        >>> from scipy.signal import find_peaks
        >>> trace = np.random.randn(1000)
        >>> trace[500] = 20  # Inject a spike
        >>> spikestest_prominence(trace, prominence=10)
        array([500])
    """
    peaks, _ = find_peaks(np.abs(data), prominence=prominence)
    return peaks


def wavelet(data, wavelet='db4', level=4, coeffs_index=-1, threshold=3.5):
    """
    Detects spike-like anomalies in a 1D signal using wavelet decomposition.

    This function decomposes the signal using discrete wavelet transform (DWT), 
    examines the specified level of detail coefficients, and identifies spikes 
    as values that exceed a multiple of the standard deviation of those coefficients.
    Detected spike locations are approximately mapped back to the original signal index space.

    Parameters:
        data (np.ndarray): 1D input signal array.
        wavelet (str): Type of wavelet to use for decomposition (default is 'db4').
        level (int): Number of decomposition levels to perform (default is 4).
        coeffs_index (int): Index of the detail coefficients to use (e.g., -1 for highest frequency cD1).
        threshold (float): Number of standard deviations above which a coefficient is considered a spike (default is 3.5).

    Returns:
        np.ndarray: Array of estimated indices in the original signal corresponding to detected spikes.

    Example:
        >>> import numpy as np
        >>> a = np.random.randn(1000)
        >>> a[149] = 7  # Inject a spike
        >>> wavelet(a, level=5, coeffs_index=-1, threshold=3.5)
        array([148])  # Example output (index may vary slightly depending on settings)
    """
    coeffs = pywt.wavedec(data, wavelet, level=level)
    detail = coeffs[coeffs_index]
    std = np.std(detail)
    spike_locs = np.where(np.abs(detail) > threshold * std)[0]

    # Approximate mapping back to original signal
    factor = len(data) / len(detail)
    approx_indices = np.round(spike_locs * factor).astype(int)
    return approx_indices
