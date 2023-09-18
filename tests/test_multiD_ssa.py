import pytest

import numpy as np
import tensorly as tl
from matplotlib import pyplot as plt
import scipy.linalg  as linalg

from src.multiD_ssa import multiD_SSA_decomp


def test_multiD_SAA():
    """test multiD SSA on 2 simple signals
    """
    # create 2 signals
    t = np.linspace(0, 4 * np.pi, 100)
    sig_1 = 2 * t.copy() + np.cos(t.copy()) * (-1)
    sig_2 = -1 * t.copy() + np.cos(t.copy()) * 1
    init_sig = [sig_1, sig_2]

    # create trajectory tensors out of it using method of "multiD_SSA_decom"
    L = 50
    cpd_rank = 10
    ssa_decomp = multiD_SSA_decomp([sig_1, sig_2], L, cpd_rank)

    # perform CPD decomposition
    ssa_decomp.decompose()

    # now vectors c[i][j], i \in 1,...,r have meaning of singular values
    # let's picture norms of this vectors
    c_norms = linalg.norm(ssa_decomp.factors[2], axis=0)
    c_norms_ordered = list(zip(range(len(c_norms)), c_norms))
    c_norms_ordered.sort(key=lambda x: x[1], reverse=True)
    c_norms = [el[1] for el in c_norms_ordered]
    norms_order = [el[0] for el in c_norms_ordered]
    fig, ax = plt.subplots()
    ax.plot(c_norms, marker='.')
    ax.grid(True)
    ax.set_title('Norms of c vectors')
    #fig.show()
    
    # group components somehow
    groups = [(0, ), (i for i in range(1, 10))]
    ssa_decomp.group_components(groups)
    # extract signals
    ssa_decomp.extract_signals()

    # vizualize components. Rows = 1d signal coordinate-wise, Columns = extracted component
    fig, ax = plt.subplots(nrows=2, ncols=len(groups), figsize=(10, 10))
    for row in range(2):
        for col in range(len(groups)):
            ax[row][col].plot(t, ssa_decomp.component_signals[col][row])
            ax[row][col].grid(True)
            ax[row][col].set_title(f"Signal #{row + 1}, component #{col + 1}")
    
    #fig.show()

    # visualize intial and restored components
    fig, ax = plt.subplots(nrows=2)
    for row in range(2):
        cur_restored_sig = np.zeros(t.size)
        for col in range(len(groups)):
            cur_restored_sig += ssa_decomp.component_signals[col][row]

        ax[row].plot(t, cur_restored_sig, label='restored')
        ax[row].plot(t, init_sig[row], label='intial', linestyle='dashed')
        ax[row].legend()
        ax[row].grid(True)
        ax[row].set_title(f"Signal #{row + 1}")

    fig.show()
    pass
    # we can see rather good results

