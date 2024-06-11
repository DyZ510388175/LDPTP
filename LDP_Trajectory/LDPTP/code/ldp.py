import json

import numpy as np
import random
import math

from scipy.optimize import minimize

from LDPTP.code.grid import is_adjacent_grids
from LDPTP.code.parse import get_args
from typing import Tuple, List


class LDPServer:
    def __init__(self, epsilon, grid_map):
        """
        General class of server side
        :param epsilon: privacy budget
        :param map: grid list
        :param map_func: index mapping function
        """
        self.epsilon = epsilon

    def aggregate(self, data):
        """
        Aggregate users' updated data items
        :param data: real data item updated by user
        """
        raise NotImplementedError('Aggregation on sever not implemented!')

    def adjust(self):
        """
        Adjust aggregated data to get unbiased estimation
        """
        raise NotImplementedError('Adjust on sever not implemented!')

    def initialize(self, epsilon, d, map_func=None):
        self.epsilon = epsilon
        self.d = d
        # self.map_func = lambda x: x if (map_func is None) else map_func
        self.map_func = map_func

        # Sum of updated data
        self.aggregated_data = np.zeros(self.d)
        # Adjusted from aggregated data
        self.adjusted_data = np.zeros(self.d)

        # Number of users
        self.n = 0


class LDPClient:
    def __init__(self, epsilon, grid_map, time_parts):
        """
        General class of client side
        :param epsilon: privacy budget
        :param map: grid list
        :param map_func: index mapping function
        """
        self.epsilon = epsilon

    def _perturb(self, index):
        """
        Internal method for perturbing real data
        :param index: index of real data item
        """
        raise NotImplementedError('Perturb on client not implemented!')

    def privatise(self, data):
        """
        Public method for privatising real data
        :param data: data item
        """
        raise NotImplementedError('Privatise on sever not implemented!')

    def initialize(self, epsilon, grid_map):
        self.epsilon = epsilon
        self.d = grid_map.size


"""---------------OUE---------------"""


class OUEServer(LDPServer):
    def __init__(self, epsilon, d, map_func=None):
        """
        Optimal Unary Encoding of server side
        """
        super(OUEServer, self).__init__(epsilon, d, map_func)

        # Probability of 1=>1
        self.p = 0.5
        # Probability of 0=>1
        self.q = 1 / (math.pow(math.e, self.epsilon) + 1)
        # self.q = 0

        # Number of users
        self.n = 0

    def aggregate(self, data):
        self.aggregated_data += data
        self.n += 1

    def adjust(self) -> np.ndarray:
        # Real data, don't adjust
        if self.epsilon < 0:
            self.adjusted_data = self.aggregated_data
            return self.adjusted_data

        self.adjusted_data = (self.aggregated_data - self.n * self.q) / (self.p - self.q)
        return self.adjusted_data

    def estimate(self, data) -> float:
        """
        Estimate frequency of a specific data item
        :param data: data item
        :return: estimated frequency
        """
        index = self.map_func(data)
        return self.adjusted_data[index]


class OUEClient(LDPClient):
    def __init__(self, epsilon, d, map_func=None):
        """
        Optimal Unary Encoding of server side
        """
        super(OUEClient, self).__init__(epsilon, d, map_func)

        # Probability of 1=>1
        self.p = 0.5
        # Probability of 1=>1
        self.q = 1 / (math.pow(math.e, self.epsilon) + 1)
        # self.q = 0

    def _perturb(self, index):
        # Remember that p is the probability for 1=>1;
        # And q is the probability for 0=>1

        # Update real data
        if self.epsilon < 0:
            perturbed_data = np.zeros(self.d)
            perturbed_data[index] = 1
            return perturbed_data

        # If y=0, Prob(y'=1)=q, Prob(y'=0)=1-q
        perturbed_data = np.random.choice([1, 0], size=self.d, p=[self.q, 1 - self.q])

        # If y=1, Prob(y'=0)=p
        if random.random() < self.p:
            perturbed_data[index] = 1
        else:
            perturbed_data[index] = 0

        return perturbed_data

    def privatise(self, grid_map, point_gridAndtime):
        index = self.map_func(point_gridAndtime)
        return self._perturb(index)


"""------------GRR---------------------------------------------------"""


class GRRServer(LDPServer):
    def __init__(self, epsilon, grid_map):
        """
        Optimal Unary Encoding of server side
        """
        super(GRRServer, self).__init__(epsilon, grid_map)

        self.map = grid_map.map
        self.d = grid_map.size
        # Probability of 1=>1
        self.p = math.pow(math.e, self.epsilon) / (math.pow(math.e, self.epsilon) + self.d - 1)
        # Probability of 0=>1
        self.q = (1 - self.p) / (self.d - 1)
        # self.q = 0

        # Number of users
        # self.n = [0] * get_args().time_num
        self.n = np.zeros(get_args().time_num)

        # perturbed trajectories
        self.perturbed_grid_db = list()

        # Candidate regions at different times
        self.candidate_regions = [list() for _ in range(get_args().time_num)]

    def aggregate(self, perturbed_index, time_idx):
        # print("x_idx, y_idx:", perturbed_index[0], perturbed_index[1])
        self.map[perturbed_index[0]][perturbed_index[1]].point_num_Withtimes[time_idx] += 1
        self.n[time_idx] += 1

    def aggregate_t(self, perturbed_t_index):
        # perturbed_t = list()
        # for index in perturbed_t_index:
        #     perturbed_t.append(self.map[index[0]][index[1]])
        perturbed_t = [self.map[index[0]][index[1]] for index in perturbed_t_index]
        self.perturbed_grid_db.append(perturbed_t)

    def adjust_and_generate_candidate_region(self) -> list(list()):
        # Real data, don't adjust
        if self.epsilon < 0:
            return
        # print("density_threshold:", density_threshold)
        for row_grids in self.map:
            for grid in row_grids:
                for i, point_num in enumerate(grid.point_num_Withtimes):
                    density_threshold = (get_args().mu * np.sqrt(self.n[i])) / self.epsilon
                    # print("grid.point_num_Withtimes[i], density_threshold:", grid.point_num_Withtimes[i], density_threshold)

                    # with open(f'../data/{get_args().dataset}_parameter_stats.json', 'w') as f:
                    #     json.dump({'density_threshold': density_threshold}, f)

                    # print("grid.point_num_Withtimes[i], self.n[i],get_args().mu:", grid.point_num_Withtimes[i], self.n[i],get_args().mu)
                    # print("self.n, self.p, self.q:", self.n, self.p, self.q)
                    # grid.point_num_Withtimes[i] = (grid.point_num_Withtimes[i] - self.n[i] * self.q) / (self.p - self.q)
                    if point_num < density_threshold:
                        # print("grid.point_num_Withtimes[i],density_threshold:", grid.point_num_Withtimes[i],
                        #       density_threshold)
                        # if grid.point_num_Withtimes[i] > 0:
                        #     print("grid.point_num_Withtimes[i],density_threshold:", grid.point_num_Withtimes[i],density_threshold)
                        grid.point_num_Withtimes[i] = 0
                    else:
                        if grid not in self.candidate_regions[i]:
                            # print("self.candidate_regions[i]:", self.candidate_regions[i])
                            self.candidate_regions[i].append(grid)
                    # print("grid.point_num_Withtimes[i]2:", grid.point_num_Withtimes[i])

    def get_candidate_regions(self):
        return self.candidate_regions

    def get_perturbed_grid_db(self):
        return self.perturbed_grid_db

    def estimate(self, data) -> float:
        """
        Estimate frequency of a specific data item
        :param data: data item
        :return: estimated frequency
        """
        index = self.map_func(data)
        return self.adjusted_data[index]

    def find_timeIndex_for_minute(self, time, time_parts):
        for idx, (start_minute, end_minute) in enumerate(time_parts):
            if start_minute <= time <= end_minute:
                # print("time,(start_minute, end_minute):", time, (start_minute, end_minute))
                return idx
        return None


class GRRClient(LDPClient):
    def __init__(self, epsilon, grid_map, time_parts):
        """
        Optimal Unary Encoding of server side
        """
        super(GRRClient, self).__init__(epsilon, grid_map, time_parts)

        # Probability of 1=>1
        self.d = grid_map.size
        # self.d = 2
        self.p = math.pow(math.e, self.epsilon) / (math.pow(math.e, self.epsilon) + self.d - 1)
        # self.grid_map = grid_map
        # Probability of 1=>1
        self.q = (1 - self.p) / (self.d - 1)
        # self.q = 0

    def _perturb(self, mapped_grid_index):
        # Remember that p is the probability for 1=>1;
        # And q is the probability for 0=>1
        # Update real data
        if self.epsilon < 0:
            return mapped_grid_index

        # If y=0, Prob(y'=1)=q, Prob(y'=0)=1-q
        if np.random.random() < self.p:
            return mapped_grid_index

        grid_index = mapped_grid_index[0] * get_args().grid_num + mapped_grid_index[1]
        # perturb_domain = [i for i in range(0, self.d)]
        # # print("args.grid_num, grid_index,perturb_domain:", get_args().grid_num, grid_index,perturb_domain)
        # perturb_domain.remove(grid_index)
        # perturbed_index = perturb_domain[np.random.randint(low=0, high=len(perturb_domain))]

        perturb_domain = np.delete(np.arange(self.d), grid_index)
        perturbed_index = np.random.choice(perturb_domain)

        return (perturbed_index // get_args().grid_num, perturbed_index % get_args().grid_num)

    def privatise(self, point_gridAndtime):
        # print("data:", point_gridAndtime)
        mapped_grid_index = point_gridAndtime[0].index
        return self._perturb(mapped_grid_index)

    def get_time_index(self, time, time_parts):
        return find_timeIndex_for_minute(time, time_parts)


"""------------EM---------------------------------------------------"""


class EMServer(LDPServer):
    def __init__(self, epsilon, candidate_region):
        """
        Optimal Unary Encoding of server side
        """
        super(EMServer, self).__init__(epsilon, candidate_region)

        self.candidate_region = candidate_region

        self.perturbed_transitions_list = list()  #: List[List[perturbed_trajectory]]
        self.perturbed_start2end_list = list()  #: List[List[perturbed_trajectory]]
        self.corrected_transitions_list = list()  # 返回索引
        self.corrected_start2end_list = list()  # 返回索引

    def aggregate(self, perturbed_trajectory_transitions):
        self.perturbed_transitions_list.append(perturbed_trajectory_transitions)

    def aggregate_start2end(self, perturbed_start2end):
        self.perturbed_start2end_list.append(perturbed_start2end)

    def transition_correction(self, grr_perturbed_trajectories):
        corrected_trajectories = list()
        print("len(grr_perturbed_trajectories):", len(grr_perturbed_trajectories))
        print("len(self.perturbed_transitions_list):", len(self.perturbed_transitions_list))
        print("len(self.perturbed_start2end_list):", len(self.perturbed_start2end_list))
        for t_index, traj in enumerate(grr_perturbed_trajectories):
            corrected_trajectory = list()
            corrected_start = None
            corrected_end = None
            traj_len = len(traj) - 1

            # if traj_len < 1:
            #     print("t_index:", t_index)
            #     print("grr_perturbed_trajectories[t_index]:", grr_perturbed_trajectories[t_index])
            #     print("self.perturbed_transitions_list[t_index]:", self.perturbed_transitions_list[t_index])
            #     print("self.perturbed_start2end_list:", self.perturbed_start2end_list[t_index])
            for g_index in range(traj_len):
                # print("trajectory_len:", len(grr_perturbed_trajectories[t_index]) - 1)
                cur_grid_grr = traj[g_index]
                next_grid_grr = traj[g_index + 1]
                t_transitions = self.perturbed_transitions_list[t_index]

                cur_grid_1_em = t_transitions[g_index][0]
                next_grid_1_em = t_transitions[g_index][1]
                if g_index == 0:
                    cur_grid_2_em = self.perturbed_start2end_list[t_index][0]
                else:
                    cur_grid_2_em = t_transitions[g_index - 1][1]
                if g_index == traj_len-1:
                    next_grid_2_em = self.perturbed_start2end_list[t_index][1]
                else:
                    next_grid_2_em = t_transitions[g_index + 1][0]

                cur_grid_index = self.get_corrected_index(
                    np.array([cur_grid_grr.index, cur_grid_1_em.index, cur_grid_2_em.index]))
                next_grid_index = self.get_corrected_index(
                    np.array([next_grid_grr.index, next_grid_1_em.index, next_grid_2_em.index]))
                # print("cur_grid_index:", cur_grid_index)
                corrected_trajectory.append((cur_grid_index, next_grid_index))

                # 起始/结束点
                if g_index == 0:
                    corrected_start = cur_grid_index
                if g_index == traj_len - 1:
                    corrected_end = next_grid_index
            corrected_trajectories.append(corrected_trajectory)
            if corrected_start is not None and corrected_end is not None:
                self.corrected_start2end_list.append((corrected_start, corrected_end))
        self.corrected_transitions_list = corrected_trajectories
        return self.corrected_transitions_list

    def get_corrected_trajectories(self):
        return self.corrected_transitions_list

    def get_corrected_start2end(self):
        return self.corrected_start2end_list

    def get_perturbed_transitions_list(self):
        return self.perturbed_transitions_list

    def get_corrected_transitions_list(self):
        return self.corrected_transitions_list

    def get_perturbed_start2end_list(self):
        return self.perturbed_start2end_list

    # def get_corrected_index(self, grid_indexs):
    #     # print("grid_indexs:", grid_indexs)
    #     def objective(grid_index):
    #         x, y = grid_index
    #         return get_args().w1 * ((x - grid_index[0]) ** 2 + (y - grid_index[1]) ** 2) + \
    #             (1 - get_args().w1) / 2 * ((x - grid_index[0]) ** 2 + (y - grid_index[1]) ** 2 + \
    #                                  (x - grid_index[0]) ** 2 + (y - grid_index[1]) ** 2)
    #
    #     initial_guess = grid_indexs[1]
    #     corrected_grid_index = minimize(objective, initial_guess, method='BFGS')
    #     return tuple(np.round(corrected_grid_index.x).astype(int))

    def get_corrected_index(self, grid_indexs):
        # print("grid_indexs:", grid_indexs)
        def objective(initial_guess, grid_indexs):
            w1 = 0.2
            w2 = (1 - w1) / 2
            loss = 0
            for i in range(len(grid_indexs)):
                x, y = initial_guess
                diff_x = x - grid_indexs[i][0]
                diff_y = y - grid_indexs[i][1]
                if i == 0:
                    loss += w1 * (diff_x ** 2 + diff_y ** 2)
                else:
                    loss += w2 * (diff_x ** 2 + diff_y ** 2)
            return loss

        initial_guess = np.mean(grid_indexs, axis=0)
        corrected_grid_index = minimize(objective, initial_guess, args=(grid_indexs), method='BFGS')
        return tuple(np.round(corrected_grid_index.x).astype(int))


class EMClient(LDPClient):
    def __init__(self, epsilon, candidate_region, time_parts):
        """
        Optimal Unary Encoding of server side
        """
        super(EMClient, self).__init__(epsilon, candidate_region, time_parts)

        # Probability of 1=>1
        # self.d = grid_map.size
        self.time_parts = time_parts

        self.candidate_region = candidate_region

        self.list_candidate_transitions = list()  #: List[List[candidate_transition]]
        for i in range(get_args().time_num):
            self.list_candidate_transitions.append(list())
            for j in range(get_args().time_num):
                self.list_candidate_transitions[i].append(list())

    def perturb(self, candidate_transitions, cur_point_gridAndtime, next_point_gridAndtime):
        transition = (cur_point_gridAndtime[0], next_point_gridAndtime[0])

        def distance_point(grid1, grid2):
            # print("grid1, grid2:", grid1, grid2)
            return math.sqrt(sum([(x - y) ** 2 for x, y in zip(grid1, grid2)]))

        def distance_transition(candidate_transition, transition):
            if len(candidate_transition) != len(transition):
                raise ValueError("Trajectories must have the same length")
            transition_distances = [distance_point(grid1.index, grid2.index) for grid1, grid2 in
                                    zip(candidate_transition, transition)]
            return math.sqrt(sum(transition_distances))

        scores = [distance_transition(candidate_transition, transition) for candidate_transition in
                  candidate_transitions]
        max_score = max(scores)
        probabilities = [math.exp(self.epsilon * (max_score - score)) for score in scores]
        probabilities_sum = sum(probabilities)
        probabilities = [prob / probabilities_sum for prob in probabilities]

        chosen_index = random.choices(range(len(candidate_transitions)), weights=probabilities, k=1)[0]

        return candidate_transitions[chosen_index]

    def find_candidate_region_index(self, cur_point_gridAndtime, next_point_gridAndtime):

        cur_time_index = find_timeIndex_for_minute(cur_point_gridAndtime[1], self.time_parts)
        next_time_index = find_timeIndex_for_minute(next_point_gridAndtime[1], self.time_parts)
        return (cur_time_index, next_time_index)

    def generate_candidate_transition(self, candidate_region_index):
        cur_time_index, next_time_index = candidate_region_index
        # print(self.list_candidate_transitions[cur_time_index][next_time_index])

        # If a candidate transition state for this time period exists
        if self.list_candidate_transitions[cur_time_index][next_time_index]:
            return self.list_candidate_transitions[cur_time_index][next_time_index]

        # 获取候选区域
        candidate_transitions = []
        temp_candidate_transitions = []
        cur_candidate_region = self.candidate_region[cur_time_index]
        next_candidate_region = self.candidate_region[next_time_index]

        if cur_candidate_region and next_candidate_region:
            for element_i in cur_candidate_region:
                for element_j in next_candidate_region:
                    if is_adjacent_grids(element_i, element_j):
                        candidate_transitions.append((element_i, element_j))
                    else:
                        temp_candidate_transitions.append((element_i, element_j))
        else:
            print("cur_candidate_region or next_candidate_region is empty!!!!")

        if not candidate_transitions:
            candidate_transitions = random.sample(temp_candidate_transitions, math.ceil(len(temp_candidate_transitions) / 3))
        self.list_candidate_transitions[cur_time_index][next_time_index] = candidate_transitions
        # print("cur_candidate_region:", cur_candidate_region)
        # print("next_candidate_region:", next_candidate_region)
        return candidate_transitions

    def get_list_candidate_transitions(self):
        return self.list_candidate_transitions


def find_timeIndex_for_minute(time, time_parts):
    for idx, (start_minute, end_minute) in enumerate(time_parts):
        if start_minute <= time <= end_minute:
            # print("time,(start_minute, end_minute):", time, (start_minute, end_minute))
            return idx
    return None
