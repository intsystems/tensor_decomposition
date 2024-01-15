""" module contains algorithms of discrete optimization to find grouping with less hankelization residual (many signals)
"""
import numpy as np
import scipy.linalg as linalg

from copy import deepcopy
from typing import Union

from ..ssa_classic import SSA_classic
from ..m_ssa import m_SSA
from ..t_ssa import t_SSA
from .ssa_classic_partition import local_search_partitioning as loc_search_classic, reset_best_results, \
                                                                    get_best_found_grouping, get_best_integral_hankel_resid, \
                                                                    build_random_initial_grouping


def local_search_partitioning(ssa_obj: Union[m_SSA, t_SSA], init_grouping_list: list, steps_limit: int=5) -> tuple:
    """local search algorithm of finding good grouping. Proceeds until the steps limit or inability to find better solution in
        vicinity

    :param Union[m_SSA, t_SSA] ssa_obj: ssa object for multidimensional signals
    :param list init_grouping_list: list of init grouping for every signal in ssa object; if some element is None,
                                                                                         then init grouping will be random
    :param int steps_limit: number of steps to proceed, defaults to 5
    :return tuple: best grouping list and mean hankel residual list for every signal
    """
    best_grouping_list = []
    best_hankel_resids_list = []

    # technical variable for mSSA
    temp = 0
    # iterating over all signals
    for i in range(len(ssa_obj.t_s_list)):
        # technical variable for mSSA
        cur_sig_len = len(ssa_obj.t_s_list[i])

        # create ssa_classic object
        ssa_clas_temp = SSA_classic(ssa_obj.t_s_list[i], ssa_obj.L)
        
        if isinstance(ssa_obj, t_SSA):
            ssa_clas_temp.weights, ssa_clas_temp._left_factors, ssa_clas_temp._right_factors = ssa_obj._get_available_factors(i)
        if isinstance(ssa_obj, m_SSA):
            ssa_clas_temp.weights, ssa_clas_temp._left_factors, ssa_clas_temp._right_factors = \
                    ssa_obj.weights, ssa_obj._left_factors, ssa_obj._right_factors[:, temp:temp + (cur_sig_len - ssa_obj.L + 1)]
            
        temp += cur_sig_len - ssa_obj.L + 1

        # debug
        print(f'Local search for signal {i + 1}')

        # use partition algorithm for 1d signal
        loc_search_classic(ssa_clas_temp, init_grouping_list[i], steps_limit)

        # save results
        best_grouping_list.append(get_best_found_grouping())
        best_hankel_resids_list.append(get_best_integral_hankel_resid())

        # reset found result in classic_partitioning variables
        reset_best_results()
 
    return best_grouping_list, best_hankel_resids_list


