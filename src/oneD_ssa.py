import numpy as np
from scipy.linalg import svd
from .basic_ssa import Basic_SSA

class SSA_decomp(Basic_SSA):
    """class for making 1d SSA decomposition of given time series
    """
    def __init__(self, signal: list, L: int):
        """

        :param list signal: initial time series
        :param int L: size of the sliding-window
        """
        super().__init__(L)
        self.time_ser = signal


    def decompose(self):
        # construct trajectory matrix
        traj_matrix = self._build_traj_matrix(self.time_ser)

        # applying svd. It already returns singular values in decsent order!
        U, s_v, V_tr = svd(traj_matrix)

        # saving singular values in descend order
        self.weights = s_v
        # saving factor vectors
        self.left_factors = U
        self.right_factors = V_tr


    def extract_signals(self):
        """extraction of time series out of final trajectory matrices. They are collected in self.component_signals
        """
        # constructing component-signals
        for group in self.grouping:
            # constructing trajectory matrix
            cur_traj_matr = np.zeros(shape=(self.left_factors[0].shape[0], self.right_factors[0].shape[0]), dtype=np.float32)

            # summing skeletones
            for ind in group:
                cur_traj_matr += self.weights[ind] * np.outer(self.left_factors.transpose()[ind], self.right_factors[ind])

            # hankelizing traj. matrix
            self._hankelize(cur_traj_matr)

            # extract 1d signal
            self.component_signals.append(self._extract_ts(cur_traj_matr))



        