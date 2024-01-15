""" module contains algorithms of discrete optimization to find grouping with less hankelization residual (one signal)
"""

import numpy as np
import scipy.linalg as linalg

from copy import deepcopy
from time import time

from ..ssa_classic import SSA_classic

# global variables for current best groupings
# best full hankelizing residual (sum of all individual residuals)
_best_integral_hankel_resid = 1e15
# best groupping for this residual
_best_found_grouping = []


def get_best_integral_hankel_resid():
    global _best_integral_hankel_resid
    return _best_integral_hankel_resid

def get_best_found_grouping():
    global _best_found_grouping
    return _best_found_grouping


def reset_best_results():
    """ forget all previous best results for grouping
    """
    global _best_integral_hankel_resid
    global _best_found_grouping

    _best_integral_hankel_resid = 1e15
    _best_found_grouping = []


def full_search_partitioning(ssa_obj: SSA_classic, cur_el=0, grouping_list:list=[]):
        """full search algorithm; check all (B_n - 1) variants of partitioning
           Recursive algorithm, depth of recursion == number of factors in decomposition

        :param int cur_el: current depth of recursion <=> current element of choice, defaults to 0
        :param list grouping_list: current grouping, defaults to []
        """
        # base situation; put element 0 in his group
        if cur_el == 0:
            grouping_list.append([cur_el, ])
            cur_el += 1

            full_search_partitioning(ssa_obj, cur_el, grouping_list)

            cur_el -= 1
            grouping_list.pop()
        # on this stage we have groupping set. Now compute residual on it
        elif cur_el == len(ssa_obj.weights):
            # some debug
            #print(f'Group {grouping_list} is built')

            # we do not accept situation with one group (trivial)
            if len(grouping_list) != 1:
                ssa_obj.set_factors_grouping(grouping_list)
                _, hankel_resids, _ =  ssa_obj.decompose_signal()
                hankel_integral_resid = np.mean(hankel_resids)

                global _best_integral_hankel_resid
                global _best_found_grouping

                if hankel_integral_resid < _best_integral_hankel_resid:
                    _best_integral_hankel_resid = hankel_integral_resid
                    _best_found_grouping = grouping_list.copy()
        else:
            # number of groups on the current stage
            cur_num_groups = len(grouping_list)

            # situation when we put current element in one of the existing groups
            for group_num in range(cur_num_groups):
                grouping_list[group_num].append(cur_el)
                cur_el += 1

                full_search_partitioning(ssa_obj, cur_el, grouping_list)

                cur_el -= 1
                grouping_list[group_num].pop()

            # situation when we put current element in its own group
            grouping_list.append([cur_el, ])
            cur_el += 1

            full_search_partitioning(ssa_obj, cur_el, grouping_list)

            grouping_list.pop()
            cur_el -= 1

        return


def local_search_partitioning(ssa_obj: SSA_classic, init_grouping: list=None, steps_limit: int=10):
    """local search algorithm of finding good grouping. Proceeds until the steps limit or inability to find better solution in
        vicinity

    :param SSA_classic ssa_obj: _description_
    :param list init_grouping: starting point, defaults to None <=> random initial grouping
    :param int steps_limit: number of steps to proceed, defaults to 10
    """
    num_elements = len(ssa_obj.weights)

    # build random initial grouping if not given
    if init_grouping == None:
        init_grouping = build_random_initial_grouping(num_elements)

    # use initial grouping to obtain current score
    ssa_obj.set_factors_grouping(init_grouping)
    _, hankel_resids, _ = ssa_obj.decompose_signal()
    cur_score = np.mean(hankel_resids)
    cur_grouping = init_grouping

    # variables of best solution
    best_score = cur_score
    best_grouping = deepcopy(cur_grouping)

    # traveling by vicinities until limit of steps or inability to find better solution in vicinity
    for step_num in range(steps_limit):
        # debug
        print(f'Iteration {step_num + 1}')

        for group_num in range(len(cur_grouping)):
            # debug
            time_left = time()
            
            for el in cur_grouping[group_num]:
                new_grouping = deepcopy(cur_grouping)

                # put current element in some other group
                if len(cur_grouping[group_num]) > 1:
                    new_grouping[group_num].remove(el)
                else:
                    new_grouping.remove(new_grouping[group_num])

                for other_group_num in range(len(new_grouping)):
                    # skip the same group if it exist
                    if len(new_grouping) == len(cur_grouping) and other_group_num == group_num:
                        continue

                    new_grouping[other_group_num].append(el)

                    ssa_obj.set_factors_grouping(new_grouping)
                    _, local_hankel_resids, _ =  ssa_obj.decompose_signal()
                    local_score = np.mean(local_hankel_resids)

                    if local_score < best_score:
                        best_score = local_score
                        best_grouping = deepcopy(new_grouping)

                    new_grouping[other_group_num].pop()

                # put current element in own group if its group contains not only him
                if len(cur_grouping[group_num]) > 1:
                    new_grouping.append([el, ])

                    ssa_obj.set_factors_grouping(new_grouping)
                    _, local_hankel_resids, _ =  ssa_obj.decompose_signal()
                    local_score = np.mean(local_hankel_resids)

                    if local_score < best_score:
                        best_score = local_score
                        best_grouping = deepcopy(new_grouping)

                    new_grouping.pop()

                # debug
                print(f'Moved element {el}; consumed time = {time() - time_left}')
                time_left = time()

        # in case no better solutions in 1-vicinity we break
        if best_score == cur_score:
            break
        else:
            cur_score = best_score
            cur_grouping = deepcopy(best_grouping)

    global _best_integral_hankel_resid
    global _best_found_grouping

    if best_score < _best_integral_hankel_resid:
        _best_integral_hankel_resid = best_score
        _best_found_grouping = best_grouping


def build_random_initial_grouping(num_elements):
    # build random initial grouping 
    init_grouping = [[0, ]]

    # for every next element we decide which group it goes
    for cur_el in range(1, num_elements):
        num_groups = len(init_grouping)
        # num of group to go or own group
        cur_choice = np.random.randint(0, num_groups + 1)

        # own group situation
        if cur_choice == num_groups:
            init_grouping.append([cur_el, ])
        else:
            init_grouping[cur_choice].append(cur_el)

    return init_grouping