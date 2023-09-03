import tensorly as tl
import numpy as np

class Traj_Tensor_Decomp:
    """class which only manipulate with given tensor regarding it as trajectory 3d-tensor: making tensor
    decomposition, grouping and hankelizing. Finally a list of decomposed components of initial tensor is an output
    """
    def __init__(self, tr_tensor: tl.tensor):
        self.tr_t = tr_tensor


    def _CPD_decomp(self, rank):
        """method decompose traj. tensor and save weights and factors of decomposition

        :param _type_ rank: rank to which make CPD
        """
        # make CPD
        weights_temp, factors = tl.decomposition.parafac(self.tr_t, rank, normalize_factors=True)
        # getting sorted order of weights values
        temp = [(el, ind) for ind, el in enumerate(weights_temp)]
        temp.sort(key=lambda x: x[0], reverse=True)
        # var containg indexes of weights sorted descently
        ordered_ind = [el[1] for el in temp]

        # saving variables
        self._weights = weights_temp
        self._weights_order = ordered_ind

        self.factors = []
        for r in range(rank):
            # constructing a factor as an outer product of 3 vectors
            cur_factor = tl.tenalg.outer(factors.transpose()[0], factors.transpose()[1])
            cur_factor = tl.tenalg.outer(cur_factor, factors.transpose()[2])
            self.factors.append(weights_temp[r] * cur_factor)


    def _grouping_components(self, ind_groups: list):
        """method to group chosen factors into chosen groupes (sum all factors within each group)

        :param tuple ind_groups: list of tuples which form each group, indecies IN OREDERED list of weights of factors
        """
        new_factors = []
        for group in ind_groups:
            new_factor = tl.zeros(shape=self.factors[0].shape)
            for ind in group:
                new_factor += self.factors[self._weights_order[ind]]
            new_factors.append(new_factor)

        self.factors = new_factors


    def _hankelize_factors(self):
        """method hankelizing factors. Considers that the trajectory matrices are piled among the 3rd dimension
        """
        for factor in self.factors:
            for k in range(factor.shape[2]):
                factor[:,:,k] = self._hankelize(factor[:,:,k])

    
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


class multiD_SSA_decomp:
    """class for executing SSA with many 1d signals using tensor decomposition
    """
    def __init__(self, signals: list, L: int):
        self.time_series = signals
        self.window_size = L
        # container for decomposed signals
        self.component_signals = []


    def _build_traj_tensor(self):
        # number of windowsize samples
        K = len(self.time_ser) - self.window_size + 1
        # initializing trajectory tensor
        self._traj_tensor = tl.zeros(shape=(self.window_size, K, len(self.time_series)))

        # making trajectory matricies
        for j in range(len(self.time_series)):
            cur_traj_matrix = np.empty((self.window_size, K), float)
            for i in range(K):
              cur_traj_matrix.transpose()[i] = self.time_ser[i:i + self.window_size]
            
            # piling created matricies
            self._traj_tensor[:,:,j] = cur_traj_matrix


    def extract_signals(self):
        """method uses Traj_Tensor_Decomp and extract time series from it. Components of each signal are contained as matrices 
        (number of factors, window_size) 
        """
        # creating decompositor
        tensor_decompositor = Traj_Tensor_Decomp(self._traj_tensor)
        #TODO: method-button to make the decomposition itself
        # for each given signal saves its components-series into matrix (number_of_components, window_size)
        for i in range(len(self.time_series)):
            cur_signal_decomp = []
            for factor in tensor_decompositor.factors:
                signal_component = self._extract_ts(factor[:,:,i])
                cur_signal_decomp.append(signal_component)
            self.component_signals.append(cur_signal_decomp)


    
    @staticmethod
    def _extract_ts(ar: np.ndarray):
        """method extract time series out of trajectory matrix

        :param np.ndarray ar: trajectory_matrix
        :return _type_: 1d time series
        """
        first_part = ar.transpose()[:-1]
        second_part = ar[-1]
        return np.array([first_part, second_part])
