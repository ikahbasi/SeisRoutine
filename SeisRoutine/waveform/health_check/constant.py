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


@dataclass
class RepeatedSegment:
    """
    Container describing one repeated-value segment.
    """

    start: int
    end: int
    value: float
    length: int
    is_clipped: bool


@dataclass
class RepeatedValueResult:
    """
    Container for repeated-value detection results.
    """

    indices: np.ndarray
    repeated_mask: np.ndarray
    clipped_mask: np.ndarray
    repeated_segments: list[RepeatedSegment]
    clipped_segments: list[RepeatedSegment]
    total_repeated_samples: int
    total_clipped_samples: int


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


class RepeatedValueDetector:
    """
    Detect repeated-value and clipped segments.

    Two anomaly types are supported:

    - Repeated-value segments
    - Clipped segments (repeated values close to the signal peak)
    """

    def __init__(
        self,
        min_run_length=2,
        tolerance=0.0,
        relation_to_max=0.9,
    ):
        """
        Initialize the detector.

        Parameters
        ----------
        min_run_length : int, default=2
            Minimum number of consecutive repeated samples.

        tolerance : float, default=0.0
            Maximum absolute difference between repeated samples.

        relation_to_max : float, default=0.9
            Relative threshold used to identify clipped segments.
            A repeated segment is considered clipped if

                abs(value) >= relation_to_max * max(abs(signal))
        """
        self.min_run_length = min_run_length
        self.tolerance = tolerance
        self.relation_to_max = relation_to_max

    def _detect_repeated(
        self,
        signal,
    ):
        """
        Detect repeated-value segments.
        """

        signal = np.asarray(signal)

        repeated_mask = np.zeros(
            signal.shape,
            dtype=bool,
        )

        segments = []
        start = 0
        for i in range(1, len(signal) + 1):
            if i == len(signal):
                end_of_run = True
            else:
                end_of_run = (
                    abs(signal[i] - signal[start])
                    > self.tolerance
                )
            if end_of_run:
                run_length = i - start
                if run_length >= self.min_run_length:
                    repeated_mask[start:i] = True
                    segments.append(
                        RepeatedSegment(
                            start=start,
                            end=i - 1,
                            value=float(signal[start]),
                            length=run_length,
                            is_clipped=False,
                        )
                    )

                start = i

        return repeated_mask, segments

    def _detect_clipped(
        self,
        signal,
        repeated_segments,
    ):
        """
        Identify clipped segments among repeated segments.
        """

        max_abs = np.max(np.abs(signal))

        clipped_mask = np.zeros(
            signal.shape,
            dtype=bool,
        )

        clipped_segments = []
        for segment in repeated_segments:
            if (
                abs(segment.value)
                >= self.relation_to_max * max_abs
            ):
                clipped_mask[
                    segment.start: segment.end + 1
                ] = True

                clipped_segments.append(
                    RepeatedSegment(
                        start=segment.start,
                        end=segment.end,
                        value=segment.value,
                        length=segment.length,
                        is_clipped=True,
                    )
                )

        return clipped_mask, clipped_segments

    def detect(
        self,
        signal,
    ):
        """
        Detect repeated-value and clipped segments.

        Parameters
        ----------
        signal : array_like
            Input signal.

        Returns
        -------
        RepeatedValueResult
            Detection results.
        """

        signal = np.asarray(signal)

        repeated_mask, repeated_segments = (
            self._detect_repeated(signal)
        )

        clipped_mask, clipped_segments = (
            self._detect_clipped(
                signal,
                repeated_segments,
            )
        )

        indices = np.where(
            repeated_mask | clipped_mask
        )[0]

        return RepeatedValueResult(
            indices=indices,
            repeated_mask=repeated_mask,
            clipped_mask=clipped_mask,
            repeated_segments=repeated_segments,
            clipped_segments=clipped_segments,
            total_repeated_samples=int(
                repeated_mask.sum()
            ),
            total_clipped_samples=int(
                clipped_mask.sum()
            ),
        )
