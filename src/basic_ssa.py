"""
    introducing basic class for ssa, containing common complemetery methods
"""
from abc import ABC, abstractmethod
import numpy as np

class Basic_SSA:
    @abstractmethod
    def __init__(self, L: int) -> None:
        # basic parameter for SSA
        self.window_size = L
        # sets of indicies representing groups for SSA
        self.grouping = []
        # container for decomposed signals
        self.component_signals = []


    @abstractmethod
    def decompose(self):
        """method for decomposing trajectory matrix/tensor
        """
        pass


    def group_components(self, groups: list):
        """method takes a list of tuples containing indecies of factors to be combined(summed up) together

        :param list groups: list of tuples of indecies for each group
        """
        self.grouping = groups

    @abstractmethod
    def extract_signals(self):
        """method creates time series for each grouped factors
        """
        pass


    def _build_traj_matrix(self, ts: np.ndarray) -> np.ndarray:
        """building trajectory matrix from 1d time series

        :param np.ndarray ts: 1d signal
        :return np.ndarray: trajectory matrix
        """
        # number of windowsize samples
        K = len(ts) - self.window_size + 1

        # making trajectory matrix
        traj_matrix = np.empty((self.window_size, K), float)
        for i in range(K):
            traj_matrix.transpose()[i] = ts[i:i + self.window_size]

        return traj_matrix


    @staticmethod
    def _extract_ts(ar: np.ndarray):
        """method extract time series out of trajectory matrix

        :param np.ndarray ar: trajectory_matrix
        :return _type_: 1d time series
        """
        first_part = ar.transpose()[0][:-1]
        second_part = ar[-1]
        return np.hstack((first_part, second_part))


    @staticmethod
    def _hankelize(ar: np.ndarray):
        """method to hankelize single matrix

        :param np.ndarray ar: matrix to hankelize
        """
        # above the main antidiagonal and on it
        for i in range(ar.shape[0]):
            cur_sum = 0
            for j in range(i + 1):
                cur_sum += ar[i - j][j]
            avg = cur_sum / (i + 1)

            for j in range(i + 1):
                ar[i - j][j] = avg

        # below the main antidiagonal
        for i in range(1, ar.shape[1]):
            cur_sum = 0
            j = 0
            while i + j != ar.shape[1] and ar.shape[0] - j != 0:
                cur_sum += ar[(ar.shape[0] - 1) - j][i + j]
                j += 1
            avg = cur_sum / j

            j = 0
            while i + j != ar.shape[1] and ar.shape[0] - j != 0:
                ar[(ar.shape[0] - 1) - j][i + j] = avg
                j += 1