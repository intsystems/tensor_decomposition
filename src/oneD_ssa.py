import numpy as np
from scipy.linalg import svd

class SSA_decomp:
    """class for making 1d SSA decomposition of given time series
    """
    def __init__(self, signal: list, L: int):
        """

        :param list signal: initial time series
        :param int L: size of the sliding-window
        """
        self.time_ser = signal
        self.window_size = L
        # container for decomposed signals
        self.component_signals = []


    def _build_traj_matrix(self):
        # number of windowsize samples
        K = len(self.time_ser) - self.window_size + 1

        # making trajectory matrix
        self._traj_matrix = np.empty((self.window_size, K), float)
        for i in range(K):
            self._traj_matrix.transpose()[i] = self.time_ser[i:i + self.window_size]


    def _svd_decomp(self):
        # applying svd. It already returns singular values in decsent order!
        U, s_v, V_tr = svd(self._traj_matrix)

        # saving singular values in descend order
        self._weights = s_v
        # making skeletones
        self._skeletones = []
        for ind in range(len(s_v)):
            self._skeletones.append(self._weights[ind] * np.outer(U.transpose()[ind], V_tr[ind]))


    def _group_components(self, groups: list):
        """method takes a list of tuples containing indecies of skeletones to be combined(summed up) together

        :param list groups: list of tuples of indecies of each group
        """
        new_skeletones = []
        for group in groups:
            # creating new skeletone
            new_skeletone = np.empty(shape=self._skeletones[0].shape)
            # summing elements of the group
            for ind in group:
                new_skeletone += self._skeletones[ind]
            # saving summed skeketone
            new_skeletones.append(new_skeletone)

        # updating existing list of skeletones
        self._skeletones = new_skeletones


    def _hankelize_components(self):
        """hankelizing grouped skeletones
        """
        for el in self._skeletones:
            self._hankelize(el)


    def _extract_signals(self):
        """extraction of time series out of final trajectory matrices
        """
        for el in self._skeletones:
            self.component_signals.append(self._extract_ts(el))

    
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


        