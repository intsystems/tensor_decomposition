import tensorly as tl
import numpy as np
from .basic_ssa import Basic_SSA


class multiD_SSA_decomp(Basic_SSA):
    """class for executing SSA with many 1d signals using tensor decomposition
    """
    def __init__(self, signals: list, L: int, r: int):
        """

        :param list signals: list of signals
        :param int L: window size   
        :param int r: rank of CPD decomposition, can be tuned for better results if needed
        """
        super().__init__(L)
        self.time_series = signals
        self.CPD_rank = r


    def decompose(self):
        # construct traj. matrix
        traj_matr = self._construct_traj_tensor()

        # perform CPD and safe factors
        self.factors = tl.decomposition.parafac(traj_matr, rank=self.CPD_rank)[1]


    def extract_signals(self):
        """ method stacks each multidimentional signal-component 
            of size (num_of_time_series, time_series_len) into "component_signals"
        """
        # go through each group and build component of multidim signal
        for group in self.grouping:
            K = len(self.time_series[0]) - self.window_size + 1
            # traj. tensor of component-signal
            cur_traj_tens = tl.zeros(shape=(self.window_size, K, len(self.time_series)))

            for ind in group:
                # constructing component tensor
                temp = tl.tenalg.outer([self.factors[0][0:, ind], self.factors[1][0:, ind]])
                temp = tl.tenalg.outer([temp, self.factors[2][0:, ind]])
                # adding
                cur_traj_tens += temp

            # hankalizing each slice through 3rd dim
            for i in range(cur_traj_tens.shape[2]):
                self._hankelize(cur_traj_tens[0:, 0:, i])

            # extracting multiD signal
            cur_component = []
            for i in range(cur_traj_tens.shape[2]):
                cur_time_ser = self._extract_ts(cur_traj_tens[0:, 0:, i])
                cur_component.append(cur_time_ser)

            self.component_signals.append(cur_component)


    def _construct_traj_tensor(self):
        # initializing traj. tensor
        K = len(self.time_series[0]) - self.window_size + 1
        traj_tens = tl.zeros(shape=(self.window_size, K, len(self.time_series)))

        # filling it with traj. matrices
        for i, t_s in enumerate(self.time_series):
            cur_traj_matrix = self._build_traj_matrix(t_s)
            traj_tens[0:, 0:, i] = cur_traj_matrix

        return traj_tens
