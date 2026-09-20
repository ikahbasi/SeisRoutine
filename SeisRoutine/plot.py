import os
import matplotlib.pyplot as plt
from obspy.imaging.cm import pqlx
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SeisRoutine.core as src
import SeisRoutine.statistics as srs
import numpy as np
import math
import inspect
import seaborn as sns
import logging


def plot_noise_models(
        fig,
        ax,
        noise_model_filepath,
        x_axes='period',
    ):

    noise_model = np.load(noise_model_filepath)
    periods=noise_model['model_periods']
    nlnm=noise_model['low_noise']
    nhnm=noise_model['high_noise']

    if x_axes=='period':
        x = periods
        x_txt = 0.06
        r_txt = -10
    elif x_axes=='frequency':
        x = 1 / periods
        x_txt = 1 / 0.06
        r_txt = +10
    kwargs = {
        'fontsize': 9,
        'color': "k",
        'fontfamily': "serif",
        'ha': "center",
        'va': "center",
    }

    ax.plot(x, nlnm, '0.4', linewidth=2, zorder=10)
    ax.plot(x, nhnm, '0.4', linewidth=2, zorder=10)

    plt.text(
        x=x_txt, y=-90, s="HNM",
        rotation=r_txt,
        **kwargs
    )
    plt.text(
        x=x_txt, y=-168, s="LNM",              
        rotation=0,
        **kwargs
    )


def _get_proper_kwargs(func, kwargs):
    """
    Filters kwargs to only include those that are valid parameters for func.

    Args:
        func: The callable function.
        kwargs: A dictionary of keyword arguments.

    Returns:
        A dictionary of filtered keyword arguments.
    """
    if not callable(func):
        raise TypeError("func must be callable.")
    sig = inspect.signature(func)
    valid_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    unused_kwargs = {k: v for k, v in kwargs.items() if k not in sig.parameters}
    if unused_kwargs:
        logging.debug(f"Warning: Unused kwargs in {func.__name__} function:\n\t{unused_kwargs}")
    return valid_kwargs


def _finalise_ax(ax, xlabel=None, ylabel=None, xlim=None, ylim=None,
                 labelsize=10, linewidth=2, grid=False, title=None,
                 xscale='linear', yscale='linear',
                 legend=False, legend_loc=None, **kwargs):
    """
    Finalizes an axes object with common formatting options.

    Args:
        ax: The axes object to finalize.
        xlabel: The label for the x-axis.
        ylabel: The label for the y-axis.
        xlim: A list or tuple of [xmin, xmax] for the x-axis limits.
        ylim: A list or tuple of [ymin, ymax] for the y-axis limits.
        labelsize: The font size for axis labels and tick labels.
        linewidth: The width of the axis spines.
        grid: Whether to show the grid.
        title: Title of the axis.
        xscale: scale of the x axis.
        yscale: scale of the y axis.
        legend: boolean to show the legend.
        **kwargs: Additional keyword arguments.
    """
    if xlim and (not isinstance(xlim, (list, tuple)) or len(xlim) != 2):
        raise ValueError("xlim must be a list or tuple of length 2.")
    if ylim and (not isinstance(ylim, (list, tuple)) or len(ylim) != 2):
        raise ValueError("ylim must be a list or tuple of length 2.")
    if labelsize <= 0 or linewidth <= 0:
        raise ValueError("labelsize and linewidth must be positive numbers.")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #
    ax.xaxis.set_tick_params(labelsize=labelsize)
    ax.yaxis.set_tick_params(labelsize=labelsize)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(linewidth)
    if grid:
        ax.grid()
    if title:
        ax.set_title(title)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if legend:
        ax.legend(loc=legend_loc)


def _finalise_figure(fig, show=True, save=False, savefile="figure.png",
                     suptitle=None, return_figure=False, figsize=(10.5, 7.5),
                     dpi=130, bbox_inches="tight", suptitle_fontsize=20, **kwargs):
    """
    Finalizes a figure object with common formatting options.

    Args:
        fig: The figure object to finalize.
        show: Whether to show the figure.
        save: Whether to save the figure.
        savefile: The path to save the figure to.
        suptitle: The title of the figure.
        return_figure: Whether to return the figure object.
        figsize: The size of the figure in inches.
        dpi: Dots per inch for saved figure.
        bbox_inches: Bounding box inches for saved figure.
        suptitle_fontsize: The font size for figure suptitle.
        **kwargs: Additional keyword arguments.
    """
    if not isinstance(figsize, tuple) or\
           len(figsize) != 2 or\
           not all(isinstance(i, (int, float)) and i > 0 for i in figsize):
        raise ValueError("figsize must be a tuple of two positive numbers.")
    fig.set_size_inches(figsize)
    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    if save:
        path = os.path.dirname(savefile)
        if path:
            os.makedirs(path, exist_ok=True)
        fig.savefig(savefile, bbox_inches=bbox_inches, dpi=dpi)
        logging.info(f"Saved figure to {os.path.abspath(savefile)}")
    if show:
        plt.show(block=True)
    if return_figure:
        return fig
    else:
        return None


def plot_density_meshgrid(x: np.array, y: np.array,
                          xstep: float, ystep: float,
                          ax=None,
                          **kwargs):
    '''
    Doc ???
    '''
    if ax is None:
        fig, ax = plt.subplots()
    #
    xcenters, ycenters, z = src.density_meshgrid(
        x=x, y=y, xstep=xstep, ystep=ystep, zreplace=0.9
    )
    # Get a proper kwargs for the plt.pcolormesh function.
    kw = _get_proper_kwargs(func=plt.pcolormesh, kwargs=kwargs)
    im = ax.pcolormesh(
        xcenters, ycenters, z,
        cmap=pqlx, shading='gouraud', **kw
    )
    cbaxes = ax.inset_axes(
        bounds=[0.8, 0.94, 0.15, 0.03],
        transform=ax.transAxes
    )
    cbar = ax.figure.colorbar(im, cax=cbaxes, orientation='horizontal')
    cbaxes.xaxis.set_ticks_position("bottom")
    cbar.ax.set_xlabel('Counts')
    #
    kw = _get_proper_kwargs(
        func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax, **kw)
    #
    kw = _get_proper_kwargs(
        func=_finalise_figure, kwargs=kwargs)
    _finalise_figure(ax.figure, **kw)


def histogram(arr, step=0.5, log=True,
              ax=None, fig=None, orientation='horizontal', around_zero=False,
              show_statistic=True, text_position=None, **kwargs):
    """
    This function plots a histogram of the input data and provides options to
        display descriptive statistics in various positions. The descriptive
        statistics include the mean, mode, standard deviation, and variance.

    Parameters:
        arr (array-like): The input array of data.
        step (float): The bin width of the histogram.
        log (bool): Whether to use a logarithmic scale for the y-axis.
        ax (matplotlib.axes.Axes, optional): The axes to plot the histogram on. If None,
            the current axes are used, or new axes are created.
        fig (matplotlib.figure.Figure, optional): The figure to plot the histogram on.
            If ax is specified, this parameter is ignored.
        orientation (str, optional): The orientation of the histogram ('horizontal' or
            'vertical'). Defaults to 'horizontal'.
        show_statistic (bool, optional): Whether to display descriptive statistics
            (mean, mode, standard deviation, variance). Defaults to True.
        text_position (str, optional): The position of the statistics text. Valid values:
            'top_right', 'top_left', 'bottom_right', 'bottom_left'. Defaults to 'top_right'.
        **kwargs: Additional keyword arguments for customizing the axes and figure.

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> data = np.random.normal(0, 1, 1000)
        >>> histogram(data, text_position='bottom_left')
    """

    if (ax is None) and (fig is None):
        fig, ax = plt.subplots()
    elif fig is None:
        fig = ax.figure
    elif ax is None:
        ax = plt.gca()
        
    #
    bins_min = math.floor(min(arr)) - step
    bins_max = math.ceil(max(arr)) + step
    #
    bins = np.arange(bins_min, bins_max, step)
    if around_zero:
        bins -= step/2
    ax.hist(arr, bins=bins,
            alpha=1, edgecolor='k', facecolor='skyblue',
            orientation=orientation, log=log, label='teeeeest!!!!!')
    #
    if show_statistic:
        mean = np.mean(arr)
        std = np.std(arr)
        mode = srs.mode(arr)
        var = np.var(arr)
        ##
        vals = [mean, mode, std, var]
        width = max(len(str(round(_,2))) for _ in vals) + 2
        textstr = (
            r'$\mu$ (Mean) = {:>{width}.2f}' '\n'
            r'$\mathit{{Mode}}$ = {:>{width}.2f}' '\n'
            r'$\sigma$ (Std) = {:>{width}.2f}' '\n'
            r'$\mathit{{Var}}$ = {:>{width}.2f}'.format(mean, mode, std, var,
                                                       width=width))
        # Position the text based on text_position
        if text_position == 'top_right':
            ha, va, x, y = 'right', 'top', 0.98, 0.96
        elif text_position == 'top_left':
            ha, va, x, y = 'left', 'top', 0.02, 0.96
        elif text_position == 'bottom_right':
            ha, va, x, y = 'right', 'bottom', 0.98, 0.04
        elif text_position == 'bottom_left':
            ha, va, x, y = 'left', 'bottom', 0.02, 0.04
        else: # Default position
            ha, va, x, y = 'right', 'top', 0.98, 0.96
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
        plt.text(x, y, textstr,
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment=va, horizontalalignment=ha, bbox=props)
    #
    kw = _get_proper_kwargs(func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax, **kw)
    #
    kw = _get_proper_kwargs(func=_finalise_figure, kwargs=kwargs)
    _finalise_figure(ax.figure, **kw)


def density_hist(x: np.array, y: np.array,
                 xstep: float, ystep: float,
                 kind: str='density', histlog: bool=True,
                 axes: object=None,
                 **kwargs):
    """
    This function generates a 2D plot, either a density plot or a scatter plot,
        along with a marginal histogram of the y-data.

    Parameters:
        x (np.array): The x-coordinates of the data points.
        y (np.array): The y-coordinates of the data points.
        xstep (float): The bin width for the density plot or scatter plot's x-axis.
        ystep (float): The bin width for the density plot or histogram's y-axis.
        kind (str, optional): The type of 2D plot to create. Can be 'density' or
            'scatter'. Defaults to 'density'.
        histlog (bool, optional): Whether to use a logarithmic scale for the y-axis
            of the marginal histogram. Defaults to True.
        axes (object, optional): A tuple or list containing two matplotlib Axes objects.
            If None, new axes are created.
        **kwargs: Additional keyword arguments to customize the plots. These arguments
            are passed to the underlying plotting functions
            (sns.scatterplot, plot_density_meshgrid, histogram, _finalise_ax, _finalise_figure).

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> x = np.random.normal(0, 1, 1000)
        >>> y = np.random.normal(0, 1, 1000)
        >>> density_hist(x, y, xstep=0.1, ystep=0.1, kind='scatter', show=True)
    """

    file_management = {'save': kwargs.get('save', False),
                       'show': kwargs.get('show', False)}
    kwargs.update(save=False, show=False)
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            sharey='row',# sharex='col',
            gridspec_kw={'width_ratios': [5, 1.2]})
        plt.subplots_adjust(bottom=0.15, hspace=0, wspace=0)
    else:
        if not isinstance(axes, (list, tuple)) or len(axes) != 2:
            raise ValueError("axes must be a tuple of two axes objects.")
        ax1, ax2 = axes
        fig = ax1.figure
    #
    if kind == 'scatter':
        kw = _get_proper_kwargs(
            func=sns.scatterplot, kwargs=kwargs
        )
        sns.scatterplot(
            x=x, y=y,
            alpha=0.4, s=20, color='black',
            ax=ax1, **kw
        )
    elif kind == 'density':
        plot_density_meshgrid(
            x, y,
            xstep=xstep, ystep=ystep,
            ax=ax1, **kwargs
        )
    kw = _get_proper_kwargs(func=_finalise_ax, kwargs=kwargs)
    _finalise_ax(ax1, **kw)
    #
    for key in ['ylabel', 'xlim', 'ylim']:
        _ = kwargs.pop(key, None)
    kwargs['xlabel'] =  'Abundance'
    histogram(
        arr=y,
        step=ystep, log=histlog,
        ax=ax2, fig=fig,
        **kwargs
    )
    kwargs.update(file_management)
    _finalise_figure(ax1.figure, **kwargs)


def picks_on_ax_of_trace(ax, picks, linestyles, color):
    '''
    DOcs {}
    '''
    ymin, ymax = ax.get_ylim()
    for key, val in picks.items():
        ax.vlines(
            x=val,
            ymin=ymin, ymax=ymax,
            label=key,
            color=color,
            linestyles=linestyles.get(key, '-.')
        )

def picks_on_station_stream(st, picks, linestyles, colors, **kwargs):
    '''
    DOcs {}
    '''
    st.normalize()
    fig = st.plot(handle=True)
    for ax in fig.axes:
        for pick_type in ['P', 'S']:
            picks_on_ax_of_trace(
                ax,
                picks=picks[pick_type],
                linestyles=linestyles,
                color=colors[pick_type]
            )
    _finalise_figure(fig, **kwargs)


def compare_multiple_distributions(*arrays, labels=None, colors=None,
                                   step=0.5, around_zero=False,
                                   text_position='top_right',
                                   **kwargs):
    num_arrays = len(arrays)

    if labels is None:
        labels = [f'Array {i+1}' for i in range(num_arrays)]
    elif len(labels) != num_arrays:
        raise ValueError("تعداد برچسب‌ها باید با تعداد آرایه‌ها برابر باشد.")

    if colors is None:
        colors = plt.cm.get_cmap('plasma', num_arrays).colors  # استفاده از رنگ‌های پیش‌فرض

    # تنظیمات نمودار
    fig = plt.figure(figsize=(10, 6))
    # sns.set_style("whitegrid")
    # plt.style.use('ggplot')

    # رسم هیستوگرام‌ها و نمایش آمار
    max_ = 0
    min_ = 0
    for arr in arrays:
        max_ = math.ceil(max(max_, arr.max())) + step
        min_ = math.floor(min(min_, arr.min()))
    bins = np.arange(min_, max_, step)
    if around_zero:
        bins -= step/2
    for i, arr in enumerate(arrays):
        counts, bins, patches = plt.hist(arr, bins=bins, alpha=0.7, label=labels[i], color=colors[i], edgecolor='k', log=True)

        stats_text = f'Mean: {np.mean(arr):.2f}\nStd: {np.std(arr):.2f}\nMedian: {np.median(arr):.2f}\nCounts: {arr.size}'
        # x = 0.05
        # y = 0.9 - i * 0.2
        props = dict(boxstyle='round', facecolor=colors[i], alpha=0.5)
        # plt.text(x, y, stats_text,
        #          transform=plt.gca().transAxes, fontsize=10,
        #          verticalalignment='top',
        #          color='k', bbox=props)
        

        if text_position == 'top_right':
            ha, va, x, y = 'right', 'top', 0.98, 0.96
        elif text_position == 'top_left':
            ha, va, x, y = 'left', 'top', 0.02, 0.96
        elif text_position == 'bottom_right':
            ha, va, x, y = 'right', 'bottom', 0.98, 0.04
        elif text_position == 'bottom_left':
            ha, va, x, y = 'left', 'bottom', 0.02, 0.04
        else: # Default position
            ha, va, x, y = 'right', 'top', 0.98, 0.96
        plt.text(x, y, stats_text,
                 transform=plt.gca().transAxes, fontsize=10,
                 color='k',
                 verticalalignment=va, horizontalalignment=ha, bbox=props)

    # حذف فریم سمت راست و بالا
    ax = plt.gca()
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # تنظیمات تکمیلی نمودار
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Comparison of Distributions')
    # plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    _finalise_ax(ax, **kwargs)
    _finalise_figure(fig, **kwargs)


# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:23:35 2026

@author: ikahbasi
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def plot_seismograms(
    datasets: Optional[Mapping[str, Dict[str, np.ndarray]]] = None,
    *,
    height_ratios: Optional[Sequence[Union[int, float]]] = None,
    time_key: str = "time",
    color_scheme: Optional[Union[str, list, dict]] = None,
    base_linewidth: float = 1.8,
    linewidth_decay: float = 0.4,
    min_linewidth: float = 0.6,
    fig_width: float = 11.0,
    panel_height: float = 2.6,
    grid: bool = True,
    **named_datasets: Dict[str, np.ndarray],
) -> tuple[plt.Figure, np.ndarray]:
    """
    Plot multi-panel earthquake time series with stacked, zero-spaced subplots
    and customizable panel height ratios.

    Parameters
    ----------
    datasets : Mapping[str, Dict[str, np.ndarray]], optional
        Dictionary mapping panel titles to signal dictionaries.
    height_ratios : Sequence[int or float], optional
        Relative height multipliers for each panel (e.g., [2, 1, 1]).
        Length must match the number of active panels.
        Defaults to None (all panels have equal height).
    time_key : str, default "time"
        The key designating the time vector in each dataset dictionary.
    color_scheme : str, list, or dict, optional
        Color assignment scheme (palette name, color list, or key-to-color
                                 mapping).
    base_linewidth : float, default 1.8
        Thickness of the first plotted curve in each panel.
    linewidth_decay : float, default 0.4
        Amount to reduce line thickness for each subsequent curve in the panel.
    min_linewidth : float, default 0.6
        Hard lower bound for curve thickness.
    fig_width : float, default 11.0
        Width of the figure in inches.
    panel_height : float, default 2.6
        Reference height in inches per unit ratio (determines total figure
                                                   height).
    grid : bool, default True
        Whether to show aligned horizontal and vertical gridlines.
    **named_datasets : Dict[str, np.ndarray]
        Alternative syntax allowing panels to be passed as keyword arguments.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes
    
    Example
    -------
        import numpy as np
        import matplotlib.pyplot as plt
    
        # Generate synthetic earthquake data
        t = np.linspace(0, 10, 1000)
    
        raw_waveform = {
            "Z": np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.25 * t),
            "E": 0.8 * np.sin(2 * np.pi * 2.0 * t + 0.4) * np.exp(-0.2 * t),
            "N": 0.6 * np.cos(2 * np.pi * 1.2 * t) * np.exp(-0.15 * t),
            "time": t,
        }
    
        envelope = {
            "Env-Z": np.abs(np.sin(2 * np.pi * 1.5 * t) * np.exp(-0.25 * t)),
            "time": t,
        }
    
        phase_probability = {
            "P-Phase": 1.0 / (1.0 + np.exp(-3.5 * (t - 2.5))),
            "S-Phase": 1.0 / (1.0 + np.exp(-3.0 * (t - 5.0))),
            "time": t,
        }
    
        custom_colors = {
            "Z": "#1f77b4",
            "E": "#2ca02c",
            "N": "#9467bd",
            "P-Phase": "#d62728",
            "S-Phase": "#ff7f0e",
        }
                
        # 3 panels: First panel is 3x taller, second is 1.5x, third is 1x
        fig, axes = plot_seismograms(
            waveform=raw_waveform,
            envelope=envelope,
            phase_probability=phase_probability,
            height_ratios=[3, 1.5, 1],
            # color_scheme="tab10",
            color_scheme=custom_colors,
        )
    
        plt.show()
    """
    # Merge and filter valid datasets
    merged_data: dict[str, dict[str, np.ndarray]] = {}
    if datasets:
        merged_data.update(
            {k: v for k, v in datasets.items()
             if v is not None and len(v) > 0}
        )
    if named_datasets:
        merged_data.update(
            {
                k: v
                for k, v in named_datasets.items()
                if v is not None and len(v) > 0
            }
        )

    num_panels = len(merged_data)
    if num_panels == 0:
        raise ValueError("No valid datasets provided to plot.")

    # Validate and prepare height ratios
    if height_ratios is not None:
        if len(height_ratios) != num_panels:
            msg = (
                f"Length of 'height_ratios' ({len(height_ratios)}) must match "
                f"the number of active panels ({num_panels})."
            )
            raise ValueError(msg)
        if any(r <= 0 for r in height_ratios):
            msg = "All entries in 'height_ratios' must be positive numbers."
            raise ValueError(msg)
        ratios = list(height_ratios)
    else:
        ratios = [1.0] * num_panels

    # Scale total figure height proportionally to the normalized ratio sum
    mean_ratio = sum(ratios) / num_panels
    total_height = max(panel_height * (sum(ratios) / mean_ratio), 3.0)

    fig, axes = plt.subplots(
        nrows=num_panels,
        ncols=1,
        sharex=True,
        figsize=(fig_width, total_height),
        gridspec_kw={
            "hspace": 0.0,
            "height_ratios": ratios,
        },
    )

    if num_panels == 1:
        axes = np.array([axes])

    def resolve_color(channel_idx: int, channel_name: str) -> Any:
        if isinstance(color_scheme, dict):
            if channel_name in color_scheme:
                return color_scheme[channel_name]
            fallback_cmap = plt.get_cmap("tab10")
            return fallback_cmap(channel_idx % 10)
        elif isinstance(color_scheme, list):
            return color_scheme[channel_idx % len(color_scheme)]
        elif isinstance(color_scheme, str):
            palette = plt.get_cmap(color_scheme)
            return palette(channel_idx % getattr(palette, "N", 10))
        return plt.get_cmap("tab10")(channel_idx % 10)

    for ax_idx, (panel_name, data_dict) in enumerate(merged_data.items()):
        ax = axes[ax_idx]

        if time_key not in data_dict:
            msg = (
                f"Missing required time key '{time_key}' "
                f"in panel '{panel_name}'."
            )
            raise KeyError(msg)

        time_vec = data_dict[time_key]
        signal_keys = [k for k in data_dict.keys() if k != time_key]

        for s_idx, sig_name in enumerate(signal_keys):
            signal_data = data_dict[sig_name]
            line_width = max(
                base_linewidth - (s_idx * linewidth_decay), min_linewidth
            )
            color = resolve_color(s_idx, sig_name)

            ax.plot(
                time_vec,
                signal_data,
                label=sig_name,
                linewidth=line_width,
                color=color,
                alpha=0.92,
            )

        if grid:
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)

        # Panel title placed inside top-left
        ax.text(
            0.015,
            0.88,
            panel_name,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="semibold",
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="square,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.85,
            ),
        )

        ax.legend(
            loc="upper right",
            frameon=True,
            framealpha=0.85,
            edgecolor="none",
            fontsize=9,
            ncol=min(len(signal_keys), 4),
        )

        # Remove intermediate tick labels while keeping the bottom panel intact
        if ax_idx < num_panels - 1:
            ax.tick_params(axis="x", which="both", labelbottom=False)

    axes[-1].set_xlabel("Time (s)", fontsize=11, labelpad=8)
    axes[0].set_xlim(
        left=min(v[time_key][0] for v in merged_data.values()),
        right=max(v[time_key][-1] for v in merged_data.values()),
    )

    return fig, axes
