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
        # applying svd
        U, s_v, V_tr = svd(self._traj_matrix)

        # getting sorted order of singular values
        temp = [(el, ind) for ind, el in enumerate(s_v)]
        temp.sort(key=lambda x: x[0], reverse=True)

        # saving singular values in descend order
        self._weights = [el[0] for el in temp]
        # making skeletones
        self._skeletones = []
        for el in temp:
            ind = el[1]
            self._skeletones.append(self._weights[el[0]] * np.outer(U.transpose()[ind], V_tr[ind]))


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
        first_part = ar.transpose()[:-1]
        second_part = ar[-1]
        return np.array([first_part, second_part])


    @staticmethod
    def _hankelize(ar: np.ndarray):
        """method to hankelize single matrix

        :param np.ndarray ar: matrix to hankelize
        """
        # above the main antidiagonal and on it
        for i in range(ar.shape[0]):
            cur_sum = 0
            for j in range(i + 1):
                cur_sum += ar[i + j][j]
            avg = cur_sum / (i + 1)

            for j in range(i + 1):
                ar[i + j][j] = avg

        # below the antidiagonal
        for i in range(ar.shape[0] - 1):
            cur_sum = 0
            for j in range(i + 1):
                cur_sum += ar[ar.shape[0] + j][(ar.shape[1] - i) + j]
            avg = cur_sum / (i + 1)

            for j in range(i + 1):
                ar[(ar.shape[0] - i) + j][ar.shape[1] - i + j] = avg


        