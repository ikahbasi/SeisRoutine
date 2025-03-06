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
              ax=None, fig=None, orientation='horizontal',
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
        >>> plt.show()
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
    #
    file_management = {'save': kwargs.get('save', False),
                       'show': kwargs.get('show', False)}
    kwargs.update(save=False, show=False)
    if axes is None:
        fig, (ax1, ax2) = plt.subplots(
            1, 2,
            figsize=(12, 6),
            sharey='row',# sharex='col',
            gridspec_kw={'width_ratios': [5, 1]})
        plt.subplots_adjust(bottom=0.15, hspace=0, wspace=0)
    else:
        (ax1, ax2) = axes
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
