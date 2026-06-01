import numpy as np
from scipy import stats
import pywt


def compute_power(data,  axis=0, domain='time'):
    n = data.shape[axis]
    if domain=='time':
        p = 1/n    * np.sum(np.abs(data)**2, axis=axis)
    elif domain=='frequency':
        p = 1/n**2 * np.sum(np.abs(data)**2, axis=axis)
    return p


def _snr_time(signal, noise, epsilon=1e-8, axis=0, dB=False):
    p_signal = compute_power(signal, axis=axis, domain='time')
    p_noise  = compute_power(noise, axis=axis, domain='time')
    p_noise += epsilon # Avoid divide-by-zero
    snr = p_signal / p_noise
    if dB:
        snr = 10 * np.log10(snr)
    return snr


def _snr_freq(signal, noise, epsilon=1e-8, axis=0, dB=False):
    noise_fft = np.fft.fft(noise, axis=axis)
    signal_fft = np.fft.fft(signal, axis=axis)
    # Power (squared magnitude)
    p_signal = compute_power(signal_fft,  axis=axis, domain='frequency')
    p_noise  = compute_power(noise_fft,  axis=axis, domain='frequency')
    p_noise += epsilon # Avoid divide-by-zero    
    snr = p_signal / p_noise
    if dB:
        snr = 10 * np.log10(snr)
    return snr


def _snr_h2v(signal, noise, epsilon=1e-8, axis=0, dB=False):
    noise_fft = np.fft.fft(noise, axis=axis)
    signal_fft = np.fft.fft(signal, axis=axis)

    h2v = np.abs(signal_fft)**2/np.abs(noise_fft)**2
    # Power (squared magnitude)
    p_signal = compute_power(signal_fft,  axis=axis, domain='frequency')
    p_noise  = compute_power(noise_fft,  axis=axis, domain='frequency')
    p_noise += epsilon # Avoid divide-by-zero    
    snr = p_signal / p_noise
    if dB:
        snr = 10 * np.log10(snr)
    return snr


def compute_snr(data, pick_idx,
                noise_window=100, signal_window=200,
                domain='time', axis=1, epsilon=1e-8, dB=False):
    """
    Compute the signal-to-noise ratio (SNR) around a pick index.

    Parameters
    ----------
        data : np.ndarray
            Seismic waveform of shape (channels, samples).
        pick_idx : int
            Index of the phase pick (e.g., P or S arrival).
        noise_window : int, default=100
            Number of samples before the pick to consider as noise.
        signal_window : int, default=200
            Number of samples after the pick to consider as signal.

    Returns
    -------
        snr : np.ndarray
            SNR values per channel.
    """
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    n_channels, n_samples = data.shape
    s = max(0, pick_idx - noise_window)
    e = pick_idx
    noise = data[:, s: e]
    #
    s = pick_idx
    e = min(n_samples, pick_idx + signal_window)
    signal = data[:, s: e]
    #
    if domain == 'time':
        snr = _snr_time(signal, noise, epsilon=epsilon, axis=axis, dB=dB)
    elif domain == 'frequency':
        snr = _snr_freq(signal, noise, epsilon=epsilon, axis=axis, dB=dB)
    return snr


def compute_snr_using_mad(data, pick_idx,
                          noise_window=200, signal_window=200, axis=1):
    """
    Compute Signal-to-Noise Ratio (SNR) using Median Absolute Deviation (MAD).

    This function estimates SNR as the ratio of signal MAD to noise MAD.
    Noise and signal segments are extracted relative to a given index (`pick_idx`),
    with configurable window lengths. The function handles 1D inputs by converting
    them to 2D (1 channel). A warning is issued if the requested window lengths
    cannot be fully obtained from the data.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 1D (samples,) or 2D (channels, samples).
    pick_idx : int
        Index marking the start of the signal window. Noise window ends at this index.
    noise_window : int, default=200
        Number of samples to use as noise, taken immediately before `pick_idx`.
    signal_window : int, default=200
        Number of samples to use as signal, starting at `pick_idx`.
    axis : int, default=1
        Axis along which to compute MAD. For 2D data, axis=1 computes per-channel MAD
        (across time samples). For 1D data, this parameter is effectively ignored.

    Returns
    -------
    snr_mad : np.ndarray
        Signal-to-noise ratio computed as signal_mad / noise_mad.
        If input is 1D, returns a scalar.
        If input is 2D (channels, samples), returns a 1D array of length `n_channels`.

    Warns
    -----
    UserWarning
        If the available noise window length is less than `noise_window`, or
        the available signal window length is less than `signal_window`.

    Notes
    -----
    - Noise segment: data[:, pick_idx - noise_window : pick_idx]
    - Signal segment: data[:, pick_idx : pick_idx + signal_window]
    - MAD is computed using `scipy.stats.median_abs_deviation` with default parameters:
      `center=None`, `scale=1.0`, `nan_policy='propagate'`, `keepdims=False`.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import stats
    >>> data = np.random.randn(3, 1000)
    >>> snr = compute_snr_using_mad(data, pick_idx=500, noise_window=200, signal_window=200)
    >>> print(snr.shape)
    (3,)

    >>> # 1D input
    >>> data_1d = np.random.randn(1000)
    >>> snr_scalar = compute_snr_using_mad(data_1d, pick_idx=500)
    >>> print(type(snr_scalar))
    <class 'numpy.float64'>
    """
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    n_channels, n_samples = data.shape
    sn = max(0, pick_idx - noise_window)
    en = pick_idx
    noise = data[:, sn: en]
    #
    ss = pick_idx
    es = min(n_samples, pick_idx + signal_window)
    signal = data[:, ss: es]
    if ((en-sn) < noise_window) or ((es-ss) < signal_window):
        msg = (f"Required noise length is {noise_window}, "
               f"but only {en-sn} is available"
               "\n"
               f"Required signal length is {signal_window}, "
               "but only {es-ss} is available")
        Warning(msg)
    ### MAD
    noise_mad = stats.median_abs_deviation(
        x=noise, axis=axis, center=None, scale=1.0,
        nan_policy='propagate', keepdims=False)
    signal_mad = stats.median_abs_deviation(
        x=signal, axis=axis, center=None, scale=1.0,
        nan_policy='propagate', keepdims=False)
    snr_mad = signal_mad / noise_mad
    return snr_mad


def compute_snr_using_percentile(data, pick_idx,
                                 noise_window=200, signal_window=200,
                                 lbp=25, hbp=95, axis=1):
    """
    Compute Signal-to-Noise Ratio (SNR) using percentile-based trimming.

    This function estimates SNR per channel as the ratio of median values
    from the inter-percentile range of signal and noise segments. For each
    channel, values below the `lbp`-th percentile and above the `hbp`-th
    percentile are discarded. The median of the remaining values is computed
    for both signal and noise, and their ratio gives the SNR. This approach
    reduces the influence of outliers.

    Parameters
    ----------
    data : np.ndarray
        Input data array. Can be 1D (samples,) or 2D (channels, samples).
    pick_idx : int
        Index marking the start of the signal window. Noise window ends at this index.
    noise_window : int, default=200
        Number of samples to use as noise, taken immediately before `pick_idx`.
    signal_window : int, default=200
        Number of samples to use as signal, starting at `pick_idx`.
    lbp : int or float, default=25
        Lower bound percentile (0-100). Values below this percentile are excluded.
    hbp : int or float, default=95
        Upper bound percentile (0-100). Values above this percentile are excluded.
    axis : int, default=1
        Not used in the current implementation (the function processes each channel
        individually). Kept for API consistency with `compute_snr_using_mad`.

    Returns
    -------
    snr_percentile : list of float
        A list containing SNR values for each channel. The length of the list equals
        the number of channels (or 1 if input was 1D). Each SNR is computed as:
        median(signal[percentile_range]) / median(noise[percentile_range]).

    Warns
    -----
    UserWarning
        If the available noise window length is less than `noise_window`, or
        the available signal window length is less than `signal_window`.

    Notes
    -----
    - Noise segment: data[:, pick_idx - noise_window : pick_idx]
    - Signal segment: data[:, pick_idx : pick_idx + signal_window]
    - For each channel:
        1. Compute lower (lbp) and upper (hbp) percentiles of signal and noise.
        2. Keep only samples within [lower_percentile, upper_percentile].
        3. Compute median of the trimmed samples.
        4. SNR = median(signal_trimmed) / median(noise_trimmed).
    - This method is robust to outliers and non-Gaussian distributions.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.randn(3, 1000)
    >>> snr_list = compute_snr_using_percentile(data, pick_idx=500, 
    ...                                          noise_window=200, signal_window=200,
    ...                                          lbp=25, hbp=95)
    >>> print(len(snr_list))
    3
    >>> print(snr_list)
    [1.234, 0.987, 1.456]

    >>> # 1D input (single channel)
    >>> data_1d = np.random.randn(1000)
    >>> snr_scalar_list = compute_snr_using_percentile(data_1d, pick_idx=500)
    >>> print(snr_scalar_list[0])  # Single value in a list
    1.123

    See Also
    --------
    compute_snr_using_mad : Compute SNR using Median Absolute Deviation.
    numpy.percentile : Used to compute percentile bounds.
    numpy.median : Used to compute central tendency of trimmed data.
    """
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    n_channels, n_samples = data.shape
    sn = max(0, pick_idx - noise_window)
    en = pick_idx
    noise = data[:, sn: en]
    #
    ss = pick_idx
    es = min(n_samples, pick_idx + signal_window)
    signal = data[:, ss: es]
    if ((en-sn) < noise_window) or ((es-ss) < signal_window):
        msg = (f"Required noise length is {noise_window}, "
               f"but only {en-sn} is available"
               "\n"
               f"Required signal length is {signal_window}, "
               "but only {es-ss} is available")
        Warning(msg)
    ###
    snr_percentile = []
    for ii in range(n_channels):
        signal_1d = signal[ii, :]
        signal_lb = np.percentile(signal_1d, lbp)
        signal_ub = np.percentile(signal_1d, hbp)
        signal_selected = signal_1d[(signal_lb <= signal_1d) &
                                    (signal_1d <= signal_ub)]
        #
        noise_1d = noise[ii, :]
        noise_lb = np.percentile(noise_1d, lbp)
        noise_ub = np.percentile(noise_1d, hbp)
        noise_selected = noise_1d[(noise_lb <= noise_1d) &
                                  (noise_1d <= noise_ub)]
    #
        signal_lb2ub_median = np.median(signal_selected)
        noise_lb2ub_median = np.median(noise_selected)
        snr = signal_lb2ub_median / noise_lb2ub_median
        snr_percentile.append(snr)
    return snr_percentile


def compute_snr_using_cwt(data, sps, pick_idx,
                          scales=np.arange(1, 256), wavelet='morl',
                          noise_window=100, signal_window=200,
                          noise_lag=0, signal_lag=0):
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    n_channels, n_samples = data.shape
    sn = max(0, pick_idx - noise_window) + noise_lag
    en = pick_idx + noise_lag
    ss = pick_idx + signal_lag
    es = min(n_samples, pick_idx + signal_window) + signal_lag
    snr_lst = []
    for ii in range(n_channels):
        data_channel = data[ii, :]
        coefficients, frequencies = pywt.cwt(
            data=data_channel,
            scales=scales,
            wavelet=wavelet,
            sampling_period=1/sps
        )
        energy = np.abs(coefficients)
        global_energy = np.mean(energy, axis=0)
        noise_energy =  global_energy[sn: en]
        signal_energy = global_energy[ss: es]
        snr_cwt = np.mean(signal_energy) / np.mean(noise_energy)
        snr_lst.append(snr_cwt)
    return snr_cwt


def _flat_check(data, threshold=1e-6, axis=1):
    abs_max = np.nanmax(np.abs(data), axis=axis)
    flat = abs_max < threshold
    return flat


def _variance_check(data, threshold=1e-5, axis=1):
    std = np.nanstd(data, axis=axis)
    return std < threshold


def _inf_check(data, axis=1):
    infs = np.isinf(data)
    loc_infs = np.where(infs)
    num_infs = np.sum(infs, axis=axis)
    return num_infs, loc_infs


def _nan_check(data, axis=1):
    nans = np.isnan(data)
    loc_nans = np.where(nans)
    num_nans = np.sum(nans, axis=axis)
    return num_nans, loc_nans


def is_waveform_healthy(data, axis=1, max_thr=1e-6, std_thr=1e-5):
    """
    Check the health of seismic waveform data based on basic signal integrity metrics.

    This function evaluates whether the input data is considered "healthy" using the following checks:
    - No NaN values
    - No infinite values
    - Not flat (maximum absolute amplitude above `max_thr`)
    - Not low variance (standard deviation above `std_thr`)

    Parameters
    ----------
    data : np.ndarray
        The waveform array to evaluate. Typically of shape (channels, samples).
    axis : int, default=1
        The axis along which to compute signal statistics (e.g., 1 for time dimension).
    max_thr : float, default=1e-6
        Threshold for flat signal detection. If the max absolute value is below this, it's flagged as flat.
    std_thr : float, default=1e-5
        Threshold for low-variance signal. If the standard deviation is below this, it's flagged as low-variance.

    Returns
    -------
    healthy : np.ndarray of bool
        Boolean array indicating whether each channel is healthy.
    conditions : dict
        A dictionary with the following keys, each containing per-channel results:
            - 'nan': number of NaN values
            - 'inf': number of infinite values
            - 'flat': True if signal is flat
            - 'low_variance': True if signal has low variance

    Example
    -------
    >>> import numpy as np
    >>> data = np.random.randn(3, 1000)  # 3-channel waveform
    >>> data[1] = 0  # Flat signal
    >>> data[2, 500] = np.inf  # Corrupted sample
    >>> healthy, conditions = is_waveform_healthy(data)
    >>> print(healthy)
    [ True False False]
    >>> print(conditions['flat'])
    [False  True False]
    >>> print(conditions['inf'])
    [0 0 1]
    """
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    num_nans, loc_nans = _nan_check(data, axis=axis)
    num_infs, loc_infs = _inf_check(data, axis=axis)
    flat = _flat_check(data, threshold=max_thr, axis=axis)
    var = _variance_check(data, threshold=std_thr, axis=axis)
    ###
    conditions = {'nan': num_nans,
                  'inf': num_infs,
                  'flat': flat,
                  'low_variance': var,
                }
    # Combine into a single healthy flag
    healthy = (
        (conditions['nan'] == 0) &
        (conditions['inf'] == 0) &
        (~conditions['flat']) &
        (~conditions['low_variance'])
    )

    return healthy, conditions
