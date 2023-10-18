from .multiD_ssa import multiD_SSA_decomp
from .oneD_ssa import SSA_decomp
import numpy as np

class multiD_sigwise_SSA_decomp(multiD_SSA_decomp):
    """ equal to multiD_SSA but after CP decompositoin we group and decompose each 1d signal
        individually (but not the whole tensor like in 'multiD_SSA_decomp')
    """
    def __init__(self, signals: list, L: int, r: int):
        super().__init__(signals, L, r)

    
    def decompose(self):
        """_summary_
        """
        # make CPD, save factors
        super().decompose()

        # let's sort 'singular' values for each 1d signal and save index correspondance
        self.indicies_sorted = np.flip(np.argsort(self.factors[2], axis=1), axis=1)


    def group_components(self, group_list: list):
        """save groupping for EACH 1d signal

        :param list group_list: list of groupping for each signal. Each grouping is a list of chosen groups.
                                Indices in groupping correspond to the sorted list of 'singular' values
        """
        self.grouping_list = group_list


    def extract_signals(self):
        # list of decompositions for each signal
        self.component_signals = []

        # foreach signal get grouping
        for sig_num, grouping in enumerate(self.grouping_list):
            # transform indecies from sorted array to initial indicies
            # make lambda to transform one group
            def to_initial(group):
                return tuple(map(lambda el: self.indicies_sorted[sig_num][el], group))
            
            # obtain groupping in initial indices
            init_grouping = list(map(to_initial, grouping))

            # now use 1d SSA decomposer
            decomposer = SSA_decomp([], self.window_size)
            decomposer.grouping = init_grouping
            decomposer.left_factors = self.factors[0]
            decomposer.right_factors = self.factors[1].transpose()
            decomposer.weights = self.factors[2][sig_num]
            
            # decompose and save
            decomposer.extract_signals()
            self.component_signals.append(decomposer.component_signals)
