import numpy as np


def _snr_time(signal, noise, epsilon=1e-8, axis=0):
    signal_power = np.sqrt(np.mean(signal**2, axis=axis))
    noise_power = np.sqrt(np.mean(noise**2, axis=axis))
    noise_power += epsilon # Avoid divide-by-zero
    snr = signal_power / noise_power
    return snr


def _snr_freq(signal, noise, epsilon=1e-8, axis=0):
    noise_fft = np.fft.rfft(noise, axis=axis)
    signal_fft = np.fft.rfft(signal, axis=axis)
    # Power (squared magnitude)
    noise_power = np.mean(np.abs(noise_fft)**2, axis=axis)
    noise_power += epsilon # Avoid divide-by-zero
    signal_power = np.mean(np.abs(signal_fft)**2, axis=axis)        
    snr = signal_power / noise_power
    return snr


def compute_snr(data, pick_idx,
                noise_window=100, signal_window=200,
                domain='time', axis=1, epsilon=1e-8):
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
        snr = _snr_time(signal, noise, epsilon=epsilon, axis=axis)
    elif domain == 'frequency':
        snr = _snr_freq(signal, noise, epsilon=epsilon, axis=axis)
    return snr


def _flat_check(data, threshold=1e-6, axis=1):
    abs_max = np.nanmax(np.abs(data), axis=axis)
    flat = abs_max < threshold
    return flat


def _vaiance_check(data, threshold=1e-5, axis=1):
    std = np.nanstd(data, axis=axis)
    return std < threshold


def _inf_check(data, axis=1):
    infs = np.isinf(data)
    loc_infs = np.where(infs, axis=axis)
    num_infs = np.sum(infs, axis=axis)
    return num_infs, loc_infs


def _nan_check(data, axis=1):
    nans = np.isnan(data)
    loc_nans = np.where(nans, axis=axis)
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
    num_nans, loc_nans = _nan_check(data, axis=axis)
    num_infs, loc_infs = _inf_check(data, axis=axis)
    flat = _flat_check(data, threshold=max_thr, axis=axis)
    var = _vaiance_check(data, threshold=std_thr, axis=axis)
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
