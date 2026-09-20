import numpy as np
import SeisRoutine.plot as srp

import numpy as np
import pandas as pd


def plot_generator(
    generator,
    n=None,
    target_keys=None,
    **kwargs,
):
    """
    Plot sample waveforms and phase probabilities directly from a dataset
    generator.
    """

    if generator is None:
        raise ValueError("A valid 'generator' must be provided.")

    if n is None:
        n = np.random.randint(len(generator))
    print(f"{n=}")

    # Display metadata
    metadata_row = generator.dataset.metadata.iloc[n]
    if target_keys:
        with pd.option_context("display.max_rows", None):
            print(metadata_row[target_keys])
    else:
        print(metadata_row)

    sample = generator[n]

    # Time vector construction (guaranteed matching length)
    delta = 1.0 / generator.dataset.data_format["sampling_rate"]
    
    # Check if shape is (channels, time_steps) or (time_steps, channels)
    x_data = sample["X"]
    if x_data.shape[0] == len(generator.dataset.component_order):
        n_pts = x_data.shape[1]
    else:
        # If transposed: (time_steps, channels) -> transpose to (channels, time_steps)
        x_data = x_data.T
        n_pts = x_data.shape[1]

    time = np.linspace(0, (n_pts - 1) * delta, n_pts)

    # Build panel dictionaries
    waveform = {
        k: v for k, v in zip(generator.dataset.component_order, x_data)
    }
    waveform["time"] = time

    # Phase probabilities
    y_data = sample["y"]
    if y_data.shape[0] != 3 and y_data.shape[1] == 3:
        y_data = y_data.T

    probs = {
        k: v for k, v in zip(["N", "P", "S"], y_data)
    }
    probs["time"] = time

    # Forward any custom options (e.g., height_ratios, color_scheme) via **kwargs
    fig, axes = srp.plot_seismograms(
        Waveform=waveform,
        Probability=probs,
        **kwargs,
    )

    return fig, axes
