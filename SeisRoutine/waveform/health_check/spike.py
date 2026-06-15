import numpy as np
from scipy.signal import find_peaks
import pywt
import SeisRoutine.utils.statistics as srus
from scipy.stats import skew
from scipy.stats import kurtosis


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


def differential(data, dt=0.01, threshold=5.0):
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
    diffs = np.abs(np.diff(data)) / dt
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


def variance(data,
             start_idx_noise=0,
             end_idx_noise=-1,
             threshold=5):
    """
    Detect whether a spike is present in the noise segment based on variance.

    This function extracts a noise segment from the input data using the given
    start and end indices, computes its variance and maximum absolute value,
    and compares the maximum to a threshold multiple of the variance. If the
    maximum exceeds the threshold times the variance, a spike is flagged.

    Parameters
    ----------
    data : array-like
        Input data array (e.g., NumPy array or PyTorch tensor) supporting
        slicing, `.var()`, `.item()`, and `np.abs()`.
    start_idx_noise : int, optional
        Start index for the noise segment (default is 0).
    end_idx_noise : int, optional
        End index for the noise segment (default is -1, meaning last element).
    threshold : float or int, optional
        Multiplier for the variance to set the spike detection threshold
        (default is 5).

    Returns
    -------
    spike : bool
        True if `max(|noise|) > threshold * variance(noise)`, otherwise False.
    """
    noise = data[start_idx_noise: end_idx_noise]
    var = noise.var().item()
    max = np.abs(noise).max()
    spike = False
    if max > threshold*var:
        spike = True
    return spike


def hampel(data, window_size=161, n_sigmas=3):
    """
    Detect spikes using Hampel filter.
    Based on:
        INSTANCE - the Italian seismic dataset for machine learning 
        https://doi.org/10.5194/essd-13-5509-2021
    This function was generated using ChatGPT.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal (1D array)
    window_size : int
        Sliding window size (must be odd, e.g. 161)
    n_sigmas : float
        Threshold in terms of MAD (typical: 3)

    Returns
    -------
    spikes : np.ndarray (bool)
        Boolean mask where True indicates spikes
    filtered : np.ndarray
        Signal with spikes optionally replaced by median
    """
    data = np.asarray(data)
    n = len(data)
    half_window = window_size // 2

    spike_mask = np.zeros(n, dtype=bool)
    filtered = data.copy()

    for i in range(n):
        start = max(0, i-half_window)
        end = min(i+half_window+1, n)

        window = data[start:end]
        median = np.median(window)
        mad = np.median(np.abs(window - median))

        if mad == 0:
            continue

        k = 1.4826  # MAD to standard deviation conversion factor
        threshold = n_sigmas * k * mad  # scale factor for Gaussian noise

        if np.abs(data[i] - median) > threshold:
            spike_mask[i] = True
            filtered[i] = median  # optional replacement
    spike_idx = np.where(spike_mask)[0]
    return spike_idx, filtered


def spike_by_skewness(data, threshold=5, axis=1, preprocessing=False):
    """
    Detect potential seismic spikes using the skewness of the amplitude
    distribution.

    Skewness measures the asymmetry of the data distribution. Seismic traces
    containing one or a few large-amplitude spikes typically exhibit a highly
    skewed distribution because the extreme values distort the statistical
    balance of the samples.

    Parameters
    ----------
    data : array-like
        One- and multi-dimensional seismic trace or amplitude samples.
    threshold : float, default=5
        Minimum skewness value required to classify the trace as containing
        potential spikes.

    Returns
    -------
    bool
        True if the computed skewness is greater than `threshold`, otherwise
        False.

    Notes
    -----
    Interpretation of skewness values:

    - |skewness| < 0.5:
      Approximately symmetric distribution. Typically indicates normal
      seismic behavior without significant spikes.

    - 0.5 <= |skewness| < 1:
      Mild asymmetry. May result from geological features, amplitude trends,
      or weak outliers rather than true spikes.

    - 1 <= |skewness| < 3:
      Moderate asymmetry. Can indicate the presence of outliers or localized
      high-amplitude events and may warrant further inspection.

    - 3 <= |skewness| < 5:
      Strong asymmetry. Often associated with abnormal amplitudes and possible
      spike contamination.

    - skewness >= 5:
      Very strong positive skewness. Commonly considered a strong indicator of
      one or more extreme positive-amplitude spikes in seismic data.

    Limitations
    -----------
    - This metric is sensitive to any extreme values, not only spikes.
    - Genuine seismic events with unusually large amplitudes may also produce
      high skewness values.
    - Negative spikes produce large negative skewness values and will not be
      detected by this implementation because it only checks for
      `skew(data) > threshold`.
    - For detecting both positive and negative spikes, consider using
      `abs(skew(data)) > threshold`.
    - Skewness alone is not sufficient for reliable spike detection.
      For example, a trace containing a single extreme amplitude value will
      typically produce a large skewness value and be flagged as a potential
      spike. However, traces containing both large positive and negative
      outliers may exhibit a skewness close to zero despite clearly containing
      abnormal amplitudes.

      Example:
      [0.1, 0.2, 0.3, 0.4, 12.0]
      -> High skewness, likely spike contamination.

      Example:
      [-20.0, 0.0, 0.0, 0.0, 20.0]
      -> Skewness may be close to zero even though two extreme spikes are
         present.

      Therefore, skewness should be considered a screening metric rather than
      a definitive spike detector. For improved robustness, it is recommended
      to combine skewness with additional outlier-sensitive statistics such as
      kurtosis, median absolute deviation (MAD), or modified z-score methods.

    Examples
    --------
    >>> skewness(a)
    False

    >>> skewness(trace, threshold=3)
    True
    """
    if preprocessing:
        data = np.asarray(data)
        data -= data.mean()
        data[~np.isfinite(data)] = 0

    if len(data) < 30:
        Warning("Insufficient samples for reliable skewness estimation")

    s = skew(data, bias=False, axis=axis)
    spike = abs(s) > threshold

    return spike, s


def spike_by_kurtosis(data, threshold=10, axis=1, preprocessing=False):
    """
    Detect potential seismic spikes using the kurtosis of the amplitude
    distribution.

    Kurtosis measures the heaviness of the distribution tails relative to a
    normal distribution. Seismic traces containing spikes typically exhibit
    elevated kurtosis because a small number of extreme amplitude samples
    contribute disproportionately to the fourth statistical moment.

    Parameters
    ----------
    data : array-like
        One- and multi-dimensional seismic trace or amplitude samples.
    threshold : float, default=10
        Minimum kurtosis value required to classify the trace as containing
        potential spikes.
    axis : int, default=1
        Axis along which kurtosis is computed.

    Returns
    -------
    tuple[np.ndarray | bool, np.ndarray | float]
        A tuple containing:

        - spike : bool or ndarray of bool
            True where kurtosis exceeds the specified threshold.
        - k : float or ndarray
            Computed kurtosis values.

    Notes
    -----
    Interpretation of kurtosis values (Pearson definition):

    - kurtosis ≈ 3:
      Distribution similar to a Gaussian distribution. Typically indicates
      normal seismic amplitudes without significant spikes.

    - 3 < kurtosis < 5:
      Mildly heavy tails. May indicate weak outliers or localized amplitude
      anomalies.

    - 5 <= kurtosis < 10:
      Moderate tail heaviness. Often associated with abnormal amplitudes and
      potential spike contamination.

    - 10 <= kurtosis < 20:
      Strong evidence of extreme amplitudes and likely spike presence.

    - kurtosis >= 20:
      Very heavy-tailed distribution. Commonly indicates severe spike
      contamination or acquisition artifacts.

    Limitations
    -----------
    - Kurtosis is sensitive to all extreme values, not only spikes.
    - Genuine seismic events with unusually large amplitudes may also produce
      elevated kurtosis values.
    - Kurtosis does not provide information about the polarity of anomalies.
      Positive and negative spikes contribute similarly.
    - A small number of extreme samples can dominate the metric.
    - Thresholds are data-dependent and may require calibration for different
      surveys, processing stages, or acquisition systems.

    Advantages over Skewness
    ------------------------
    - Kurtosis is sensitive to both positive and negative spikes.
    - Symmetric spike contamination can still be detected even when skewness
      is close to zero.

      Example:
      [-20.0, 0.0, 0.0, 0.0, 20.0]

      This trace may exhibit near-zero skewness due to symmetry, but its
      kurtosis will typically be elevated because of the extreme amplitudes.

    - For seismic quality control, kurtosis is often more robust than
      skewness as a first-order spike screening metric.

    Examples
    --------
    >>> spike_by_kurtosis(trace)
    (False, 3.2)

    >>> spike_by_kurtosis(trace, threshold=8)
    (True, 14.7)
    """
    if preprocessing:
        data = np.asarray(data)
        data -= data.mean()
        data[~np.isfinite(data)] = 0

    if len(data) < 30:
        Warning("Insufficient samples for reliable kurtosis estimation")

    k = kurtosis(data, fisher=False, bias=False, axis=axis)
    spike = k > threshold

    return spike, k
