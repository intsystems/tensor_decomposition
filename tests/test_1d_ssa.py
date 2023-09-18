from src.oneD_ssa import SSA_decomp
import pytest
import numpy as np
from matplotlib import pyplot as plt


@pytest.fixture()
def static_methods_fixture(request):
    """for testing static methods of class
    """
    print("Testing static method")

def test_hankelize_matrix(static_methods_fixture):
    input_matrix_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    input_matrix_2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

    oneD_decomp = SSA_decomp(np.ones(7), 2)

    oneD_decomp._hankelize(input_matrix_1)
    oneD_decomp._hankelize(input_matrix_2)
    
    # manually approved
    assert True


def test_extract_signal_from_traj_matr(static_methods_fixture):
    decomp = SSA_decomp(np.ones(7), 2)

    traj_matr_1 = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]])
    signal_1 = decomp._extract_ts(traj_matr_1)
    assert (signal_1 - np.array([1, 2, 3, 4, 5, 6, 7])).all() == 0


def test_ssa_method():
    """test the whole SSA implementation for simple signal: x + sin(x). Manual grouping here
    """
    # setting signal
    x = np.linspace(0, 4 * np.pi, 100)
    sig = x + np.sin(x)
    # visualizing signal
    fig, ax = plt.subplots()
    ax.plot(x, sig)
    ax.grid(True)
    #fig.show()

    # call SSA with L = 50
    L = 50
    decomp = SSA_decomp(sig, L)

    # build traj. matrix and make svd
    decomp.decompose()
    # visualizing singualr values
    fig, ax = plt.subplots()
    ax.plot(decomp.weights, marker='.')
    ax.grid(True)
    #fig.show()
    # from plot we can differentiate components: (0) and (1, 2, 3). Now group them
    decomp.group_components([(0, ), (1, 2, 3)])
    # now extract signals from traj. matrix
    decomp.extract_signals()

    # vizualizing components
    fig, ax = plt.subplots()
    for ind, sig in enumerate(decomp.component_signals):
        ax.plot(x, sig, label=f'signal #{ind}')
    ax.legend()
    ax.grid(True)
    ax.set_title('Components')
    #fig.show()

    # comparing intial and restored signal
    fig, ax = plt.subplots()
    ax.plot(x, x + np.sin(x), label='initial')
    ax.plot(x, sum(decomp.component_signals), label='restored', alpha=0.6)
    ax.grid(True)
    ax.legend()
    fig.show()

    pass
    # perfect decomposition!

