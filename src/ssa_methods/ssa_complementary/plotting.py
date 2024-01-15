"""plotting functions here
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

from ..ssa_classic import SSA_classic


def plot_signal(time_grid: np.ndarray, t_s: np.ndarray, figsize=None, sig_name: str='Signal', color=None):
    """generate plot of given timeseries

    :param np.ndarray t_s: given time series
    :param tuple figsize: _description_, defaults to (6,6)
    :param str sig_name: _description_, defaults to 'Signal'
    :param str color: _description_, defaults to 'orange'
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(time_grid, t_s, color=color)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$y$')
    ax.grid(True)
    ax.set_title(f'{sig_name}')

    return fig, ax


def plot_singular_values(s_v: np.ndarray, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.arange(len(s_v)), s_v, color='orange', ms=10, marker='.')

    ax.set_xlabel('Singular value number')
    ax.set_ylabel(r'$\sigma$')
    ax.grid(True)
    ax.set_title('Available singular values of trajectory matrix')

    return fig, ax


def plot_weights_of_cpd(weights: np.ndarray, figsize=None):
    fig, ax = plt.subplots(figsize=figsize)

    for i in range(weights.shape[0]):
        ax.plot(np.arange(weights.shape[1]), weights[i], ms=10, marker='.', label=f'sig {i + 1}')

    ax.set_xlabel('Weight value number')
    ax.set_ylabel(r'$c[i]$')
    ax.grid(True)
    ax.set_title(f'Available "singular" values of trajectory tensor')
    ax.legend()

    return fig, ax


def plot_component_signals(time_grid: np.ndarray, component_signals: list, sig_name: str = None, figsize=None):
    fig, axs = plt.subplots(nrows=len(component_signals), ncols=1, figsize=figsize)

    for i in range(len(component_signals)):
        axs[i].plot(time_grid, component_signals[i], label=f'Component {i + 1}')
        
        if i == 0:
            axs[i].set_ylabel('$y$')
            axs[i].set_title(f'{sig_name}')
        if i == len(component_signals) - 1:
            axs[i].set_xlabel('$t$')
        else:
            axs[i].tick_params(axis='x', labelsize=0)

        axs[i].grid(True)
        axs[i].legend()

    return fig, axs
