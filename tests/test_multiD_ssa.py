import pytest

import numpy as np
import tensorly as tl
from matplotlib import pyplot as plt

from src.multiD_ssa import multiD_SSA_decomp
from src.multiD_ssa import Traj_Tensor_Decomp

def test_building_traj_tensor():
    """teseting proper work of tensorly and method to build trajectory tensor
    """
    # create 2 signals
    x = np.linspace(0, 4 * np.pi, 100)
    sig_1 = x + np.sin(x)

    y = np.linspace(0, 4 * np.pi, 100)
    sig_2 = np.exp(y) + np.cos(y)

    # use "multiD_SSA_decom"
    L = 50
    md_ssa = multiD_SSA_decomp([sig_1, sig_2], L)
    # build traj. tensora
    md_ssa._build_traj_tensor()
    first_matr = md_ssa._traj_tensor[:, :, 0]
    second_matr = md_ssa._traj_tensor[:, :, 1]
    pass


def test_multiD_SAA():
    """test multiD SSA on 2 simple signals
    """
    # create 2 signals
    x = np.linspace(0, 4 * np.pi, 100)
    sig_1 = x + np.sin(x)

    y = np.linspace(0, 4 * np.pi, 100)
    sig_2 = np.exp(y) + np.cos(y)

    # create trajectory tensors out of it using method of "multiD_SSA_decom"
    L = 50
    ssa_decomp = multiD_SSA_decomp([sig_1, sig_2], L)
    ssa_decomp._build_traj_tensor()
    traj_tensor = ssa_decomp._traj_tensor

    # create class instance
    decomposer = Traj_Tensor_Decomp(traj_tensor)
    # apply CPD with some rank
    cpd_rank = 4
    decomposer._CPD_decomp(cpd_rank)

    # vizualize weights
    fig, ax = plt.subplots()
    ax.plot([decomposer._weights[ind] for ind in decomposer._weights_order], marker='.')
    ax.grid(True)
    #fig.show()
    
    # group components somehow
    groups = [(j, ) for j in range(4)]
    decomposer._grouping_components(groups)
    # hankelize groups
    decomposer._hankelize_factors()

    
    # vizualize initial and restored signals
    restored_signals = []
    for sig_num in range(2):
        cur_traj_matr = np.zeros((50, 51))
        # for current signal sum all component's trajectory matrices
        for factor in decomposer.factors:
            cur_traj_matr += factor[:, :, sig_num]
        cur_signal = ssa_decomp._extract_ts(cur_traj_matr)
        restored_signals.append(cur_signal)

    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].plot(x, x + np.sin(x), label='initial', linestyle='--')
    ax[0].plot(x, restored_signals[0], label='restored')
    ax[0].legend()
    ax[0].grid(True)
    ax[0].set_title('Signal 1')

    ax[1].plot(y, np.exp(y) + np.cos(y), label='initial', linestyle='--')
    ax[1].plot(y, restored_signals[1], label='restored')
    ax[1].legend()
    ax[1].grid(True)
    ax[1].set_title('Signal 2')

    fig.show()
    # we can see rather good results

