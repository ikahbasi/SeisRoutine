from dataclasses import dataclass
import numpy as np


@dataclass
class DerivativeAnomalyResult:
    """
    Container for anomaly detection results.
    """

    indices: np.ndarray
    derivative: np.ndarray
    flat_mask: np.ndarray
    spike_mask: np.ndarray


class DerivativeDetector:
    """
    Detect anomalies using the first derivative of a signal.

    Two anomaly types are supported:

    - Flat anomalies: derivative magnitude is smaller than a
      specified threshold.
    - Spike anomalies: derivative magnitude is larger than a
      specified threshold.

    The spike threshold may be estimated automatically using
    the median absolute deviation (MAD) of the derivative.
    """

    def __init__(
        self,
        dt,
        flat_threshold=1e-3,
        spike_threshold=np.inf,
        mad_multiplier=10.0,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        dt : float
            Sampling interval.
        flat_threshold : float, default=0.01
            Threshold used to identify flat samples.
        spike_threshold : float or None, default=None
            Threshold used to identify spikes. If None, the
            threshold is estimated from the data.
        mad_multiplier : float, default=10.0
            Number of MADs used when estimating the spike
            threshold.
        """
        self.dt = dt
        self.flat_threshold = flat_threshold
        self.spike_threshold = spike_threshold
        self.mad_multiplier = mad_multiplier

    def _flat_mask(
        self,
        derivative,
    ):
        """
        Create a mask identifying flat samples.

        Parameters
        ----------
        derivative : ndarray
            First derivative of the signal.

        Returns
        -------
        ndarray of bool
            Boolean mask indicating flat samples.
        """
        return np.abs(derivative) <= self.flat_threshold

    def _spike_mask(
        self,
        derivative,
    ):
        """
        Create a mask identifying spike samples.

        Parameters
        ----------
        derivative : ndarray
            First derivative of the signal.

        Returns
        -------
        ndarray of bool
            Boolean mask indicating spike samples.
        """
        abs_derivative = np.abs(derivative)

        threshold = self.spike_threshold

        if threshold is None:
            median = np.median(abs_derivative)
            mad = np.median(
                np.abs(abs_derivative - median)
            )

            if mad == 0:
                threshold = np.inf
            else:
                threshold = (
                    median
                    + self.mad_multiplier * mad
                )

        return abs_derivative >= threshold

    def _derivative(
        self,
        data,
    ):
        """
        Compute the first derivative of a signal.

        Parameters
        ----------
        data : array_like
            Input signal.

        Returns
        -------
        ndarray
            First derivative.
        """
        derivative = np.diff(data) / self.dt
        derivative = np.pad(
            derivative,
            (1, 0),
            mode="edge",
        )
        return derivative

    def detect(
        self,
        data,
    ):
        """
        Detect flat and spike anomalies.

        Parameters
        ----------
        data : array_like
            Input signal.

        Returns
        -------
        AnomalyResult
            Detection results.
        """
        derivative = self._derivative(data)

        flat_mask = self._flat_mask(
            derivative=derivative,
        )

        spike_mask = self._spike_mask(
            derivative=derivative,
        )

        indices = np.where(
            flat_mask | spike_mask
        )[0]

        result = DerivativeAnomalyResult(
            indices=indices,
            derivative=derivative,
            flat_mask=flat_mask,
            spike_mask=spike_mask,
        )

        return result
