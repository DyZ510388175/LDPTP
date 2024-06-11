import configparser
import math
import os
from typing import List, Tuple, Dict

import numpy as np

import trajectory
import ldp
from grid import GridMap, Grid
import map_func
import utils
import experiment
from experiment import SquareQuery
from parse import get_args
import dataset
import pickle
import random
import lzma

from logger.logger import ConfigParser
import multiprocessing

np.random.seed(2022)
random.seed(2022)
CORES = multiprocessing.cpu_count() // 2

config = ConfigParser(name='LDPTrace', save_dir='./')
logger = config.get_logger(config.exper_name)

logger.info(f'Parameters: {get_args()}')


# ======================= CONVERTING FUNCTIONS ======================= #


def convert_raw_to_grid(raw_trajectories: List[List[Tuple[float, float, str]]],
                        interp=True):
    # Convert raw trajectories to grid trajectories
    # grid_db = [trajectory.trajectory_point2grid(t, grid_map, interp)
    #            for t in raw_trajectories]
    grid_db = list()
    for t in raw_trajectories:
        grid_t = trajectory.trajectory_point2grid(t, grid_map, interp)
        # if len(grid_t) > 2:
        #     grid_db.append(grid_t)
        if len(grid_t) < 2:
            grid_t.append(grid_t[0])
        grid_db.append(grid_t)
    return grid_db


def convert_grid_to_raw(grid_db: List[List[Grid]]):
    raw_trajectories = [trajectory.trajectory_grid2points(g_t) for g_t in grid_db]

    return raw_trajectories


# =============================== END ================================ #


# ======================= LDP UPDATE FUNCTIONS ======================= #

def divide_day_into_parts(time_num):
    total_minutes = 24 * 60  # 一天总共的分钟数
    interval_minutes = total_minutes // time_num  # 每份的分钟数
    parts = list()
    for i in range(time_num):
        start_minute = i * interval_minutes
        end_minute = (i + 1) * interval_minutes
        # start_timestamp = (start_time + timedelta(minutes=start_minute)).strftime("%H:%M")
        # end_timestamp = (start_time + timedelta(minutes=end_minute)).strftime("%H:%M")
        # parts.append((start_timestamp, end_timestamp))
        parts.append((start_minute, end_minute))
    return parts


def estimate_candidate_region(grid_db, grr_client, grr_server, time_parts):
    """
    grid_db: List[List[Grid]]
    Return candidate space for each time period
    """
    for t in grid_db:
        # print("t:", t)
        # print("len(t):", len(t))
        perturbed_t_index = list()
        for point_gridAndtime in t:
            # mapped_gridIdx = grid_map.get_mapped_grid_idx(point_gridAndtime[0]) #point_gridAndtime (grid,time)
            perturbed_index = grr_client.privatise(point_gridAndtime)
            perturbed_t_index.append(perturbed_index)
            time_index = grr_client.get_time_index(point_gridAndtime[1], time_parts)
            # print("time_index:", time_index)
            grr_server.aggregate(perturbed_index, time_index)
        grr_server.aggregate_t(perturbed_t_index)
        # print(grid_map.map[perturbed_index[0]][perturbed_index[1]].point_num_Withtimes)
    grr_server.adjust_and_generate_candidate_region()
    # print("grr_server.n:", grr_server.n)


def estimate_max_length(grid_db: List[List[Grid]], epsilon):
    """
    Return 90% quantile of lengths
    """
    ldp_server = ldp.OUEServer(epsilon, grid_map.size, lambda x: x - 1)
    ldp_client = ldp.OUEClient(epsilon, grid_map.size, lambda x: x - 1)

    for t in grid_db:
        if len(t) > grid_map.size:
            binary_vec = ldp_client.privatise(grid_map.size)
        else:
            binary_vec = ldp_client.privatise(len(t))
        ldp_server.aggregate(binary_vec)

    ldp_server.adjust()
    sum_count = np.sum(ldp_server.adjusted_data)
    count = 0
    quantile = len(ldp_server.adjusted_data)
    for i in range(len(ldp_server.adjusted_data)):
        count += ldp_server.adjusted_data[i]
        if count >= args.max_len * sum_count:
            quantile = i + 1
            break

    return ldp_server, quantile


def server_modeling_markov(corrected_transitions_index_list):
    # [[((2, 4), (1, 5))], [((5, 3), (5, 2))
    n = args.grid_num ** 2
    one_level_mat = np.zeros((n, n), dtype=float)

    for t in corrected_transitions_index_list:
        for transition_index in t:
            grid_idx_mat = map_func.transition_index_2_mat_func(transition_index)
            # print("grid_idx_mat:", grid_idx_mat)
            one_level_mat[grid_idx_mat[0]][grid_idx_mat[1]] += 1
    # print("one_level_mat:", one_level_mat)

    # Normalize probabilities by each ROW
    markov_mat = one_level_mat / (one_level_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    # print("markov_mat:", markov_mat)
    return markov_mat


def server_start2end_probability(corrected_start2end, grid_trajectories_length):
    # corrected_start2end: [((2, 4), (1, 5)), ((5, 3), (2, 3)), ((3, 2), (1, 5)), ((1, 4), (2, 4)), ((3, 5), (5, 4))]
    # grid_trajectories_length: [2, 10, 10, 4, 7]
    server_start2end_dict = {}
    server_start2end_probability_dict = {}
    for t_idx in range(len(grid_trajectories_length)):
        cur_t_len = grid_trajectories_length[t_idx]
        if cur_t_len in server_start2end_dict:
            if corrected_start2end[t_idx] in server_start2end_dict[cur_t_len]:
                tra_idx = server_start2end_dict[cur_t_len].index(corrected_start2end[t_idx])
                server_start2end_probability_dict[cur_t_len][tra_idx] += 1
            else:
                server_start2end_dict[cur_t_len].append(corrected_start2end[t_idx])
                server_start2end_probability_dict[cur_t_len].append(1)
        else:
            server_start2end_dict[cur_t_len] = [corrected_start2end[t_idx]]
            server_start2end_probability_dict[cur_t_len] = [1]

    # server_start2end_dict: {2: [((2, 4), (1, 5))], 10: [((5, 3), (2, 3)), ((3, 2), (1, 5))], 4: [((1, 4), (2, 4))], 7: [((3, 5), (5, 4))]}
    # server_start2end_probability_dict: {2: [1], 10: [1, 1], 4: [1], 7: [1]}
    print("server_start2end_dict:", server_start2end_dict)
    print("server_start2end_probability_dict_1:", server_start2end_probability_dict)

    for key in server_start2end_probability_dict:
        total = sum(server_start2end_probability_dict[key])
        server_start2end_probability_dict[key] = [val / total for val in server_start2end_probability_dict[key]]

    print("server_start2end_probability_dict_2:", server_start2end_probability_dict)

    return server_start2end_dict, server_start2end_probability_dict


def update_markov_prob(grid_db: List[List[Grid]], epsilon, max_len=36):
    ldp_server = ldp.OUEServer(epsilon / (max_len + 1), grid_map.size * 8,
                               lambda x: x)
    ldp_client = ldp.OUEClient(epsilon / (max_len + 1), grid_map.size * 8,
                               lambda x: x)
    start_server = ldp.OUEServer(epsilon / (max_len + 1), grid_map.size,
                                 lambda x: map_func.grid_index_map_func(x, grid_map))
    start_client = ldp.OUEClient(epsilon / (max_len + 1), grid_map.size,
                                 lambda x: map_func.grid_index_map_func(x, grid_map))
    end_server = ldp.OUEServer(epsilon / (max_len + 1), grid_map.size,
                               lambda x: map_func.grid_index_map_func(x, grid_map))
    end_client = ldp.OUEClient(epsilon / (max_len + 1), grid_map.size,
                               lambda x: map_func.grid_index_map_func(x, grid_map))

    for t in grid_db:
        length = min(len(t), max_len)
        # Start point
        start = t[0]
        binary_vec = start_client.privatise(start)
        start_server.aggregate(binary_vec)
        for i in range(length - 1):
            curr_grid = t[i]
            next_grid = t[i + 1]
            if grid_map.is_adjacent_grids(curr_grid, next_grid):
                map_id = map_func.adjacent_pair_grid_map_func((curr_grid, next_grid), grid_map)
                binary_vec = ldp_client.privatise(map_id)
                ldp_server.aggregate(binary_vec)
            else:
                logger.info('Trajectory has non-adjacent moves, use non-adjacent map function!')
        end = t[length - 1]
        binary_vec = end_client.privatise(end)
        end_server.aggregate(binary_vec)

    ldp_server.adjust()
    start_server.adjust()
    end_server.adjust()
    return ldp_server, start_server, end_server


# =============================== END ================================ #


# ======================== AGGREGATE FUNCTIONS ======================= #

def generate_markov_matrix(markov_vec: np.ndarray, start_vec, end_vec):
    """
    Convert extracted Markov counts to probability matrix.
    :param markov_vec: [1 x 8n^2] numpy array
    :param start_vec: [1 x n^2] numpy array
    :param end_vec: [1 x n^2] numpy array
    :return: [n^2+1 x n^2+1] Markov probability matrix
    n^2+1th row: start -> other
    n^2+1th column: other -> end
    """
    n = grid_map.size + 1  # with virtual start and end point
    markov_mat = np.zeros((n, n), dtype=float)
    for k in range(8 * grid_map.size):
        if markov_vec[k] <= 0:
            continue

        # Find index in matrix (convert k => (i, j))
        g1, g2 = map_func.adjacent_pair_grid_inv_func(k, grid_map)

        # g2 out of bound
        if g2 is None:
            continue

        i = map_func.grid_index_map_func(g1, grid_map)
        j = map_func.grid_index_map_func(g2, grid_map)

        markov_mat[i][j] = markov_vec[k]

    for i in range(len(start_vec)):
        if start_vec[i] < 0:
            start_vec[i] = 0
        if end_vec[i] < 0:
            end_vec[i] = 0
    # Start -> other, n^2+1th row
    markov_mat[-1, :-1] = start_vec
    # Other -> end, n^2+1th column
    markov_mat[:-1, -1] = end_vec

    # Normalize probabilities by each ROW
    markov_mat = markov_mat / (markov_mat.sum(axis=1).reshape((-1, 1)) + 1e-8)
    return markov_mat


# =============================== END ================================ #


# ======================== SAMPLING FUNCTIONS ======================== #

def sample_start2end_grid_index(start2end_dict, start2end_pro, length):
    """
    N^2+1th row: virtual start -> other
    """
    prob = start2end_pro[length]

    sample_id = np.random.choice(np.arange(len(start2end_pro[length])), p=prob)
    # print("sample_id:", sample_id)
    # print("start2end_dict:", start2end_dict)
    # print("start2end_dict[length][sample_id]:", start2end_dict[length][sample_id])

    start = start2end_dict[length][sample_id][0]
    end = start2end_dict[length][sample_id][1]

    return start, end


def sample_length(length_dis: np.ndarray):
    prob = length_dis / np.sum(length_dis)

    length = np.random.choice(np.arange(len(length_dis)), p=prob)

    return length + 1


def sample_markov_next(one_level_mat: np.ndarray,
                       prev_grid_index: Grid,
                       length: int) -> Grid:
    """
    Sample next grid based on Markov probability
    :param one_level_mat: 1-level Markov matrix
    :param prev_grid: previous grid
    :return: next grid
    """
    candidates = grid_map.get_adjacent_index(prev_grid_index)
    # print("candidates:", candidates)

    candidate_probabilities = np.zeros(len(candidates) + 1, dtype=float)

    for k, (i, j) in enumerate(candidates):
        # Calculate P(Candidate|T[0 ~ k-1]) using 1-level matrix
        row = prev_grid_index[0] * args.grid_num + prev_grid_index[1]
        col = i * args.grid_num + j
        prob1 = one_level_mat[row][col]

        if np.isnan(prob1):
            candidate_probabilities[k] = 0
        else:
            candidate_probabilities[k] = prob1

    # # Virtual end point
    # row = map_func.grid_index_map_func(prev_grid, grid_map)
    # col = -1
    # prob1 = one_level_mat[row][col]

    # prob1 *= min(1.0, 0.3 + (length - 1) * 0.2)
    #
    # candidate_probabilities[-1] = prob1

    if candidate_probabilities.sum() < 0.00001:
        return prev_grid_index

    candidate_probabilities = candidate_probabilities / candidate_probabilities.sum()

    sample_id = np.random.choice(np.arange(len(candidate_probabilities)), p=candidate_probabilities)

    # End
    if sample_id == len(candidate_probabilities) - 1:
        return prev_grid_index

    return candidates[sample_id]
    # i, j = candidates[sample_id]
    # return grid_map.map[i][j]


# =============================== END ================================ #


def generate_synthetic_database(grid_trajectories_length: np.ndarray,
                                markov_mat: np.ndarray,
                                start2end_dict,
                                start2end_pro,
                                size: int):
    """
    Generate synthetic trajectories
    :param grid_trajectories_length: The length of each trajectory, [int]
    :param markov_mat: Markov matrix
    :param start2end_dict: Dict[int, List[((start_x, start_y), (end_x, end_y)),]]
    :param start2end_pro: Dict[int, List[int]]
    :param size: size of synthetic database
    """

    for i in range(len(grid_trajectories_length)):
        if grid_trajectories_length[i] < 0:
            grid_trajectories_length[i] = 0

    synthetic_db_index = list()
    for i in range(size):
        # Sample length
        length = grid_trajectories_length[i]

        # Sample start and end point
        start_index, end_index = sample_start2end_grid_index(start2end_dict, start2end_pro, length)
        # print("start_index, end_index:", start_index, end_index)

        syn_trajectory_index = [start_index]
        for j in range(1, length - 1):
            prev_grid_index = syn_trajectory_index[j - 1]
            # Sample next grid based on Markov probability
            next_grid_index = sample_markov_next(markov_mat,
                                                 prev_grid_index, len(syn_trajectory_index))
            # print("next_grid_index:", next_grid_index)
            # Virtual end point
            if next_grid_index == prev_grid_index:
                break

            syn_trajectory_index.append(next_grid_index)
        syn_trajectory_index.append(end_index)
        # print("length, len(syn_trajectory_index):", length, len(syn_trajectory_index))
        synthetic_db_index.append(syn_trajectory_index)
    # print("len(synthetic_db_index):", len(synthetic_db_index))

    # 将网格索引转为网格
    synthetic_db = [map_func.trajectory_grid_index_inv_func(t, grid_map) for t in synthetic_db_index]

    return synthetic_db


def get_start_end_dist(grid_db: List[List[Grid]]):
    dist = np.zeros(grid_map.size * grid_map.size)
    start_dist = np.zeros(grid_map.size)
    end_dist = np.zeros(grid_map.size)

    for g_t in grid_db:
        start = g_t[0]
        end = g_t[-1]
        index = map_func.pair_grid_index_map_func((start, end), grid_map)
        dist[index] += 1
        start_index = map_func.grid_index_map_func(start, grid_map)
        start_dist[start_index] += 1
        end_index = map_func.grid_index_map_func(end, grid_map)
        end_dist[end_index] += 1

    return dist, start_dist, end_dist


def get_real_density(grid_db: List[List[Grid]]):
    real_dens = np.zeros(grid_map.size)

    for t in grid_db:
        for g in t:
            index = map_func.grid_index_map_func(g, grid_map)
            real_dens[index] += 1

    return real_dens


def print_grid_trajectories_index(grid_trajectories):
    grid_trajectories_index = list()
    for t in grid_trajectories:
        # print("t:", t)
        grid_trajectory_index = list()
        for gridAndtime in t:
            # print("gridAndtime:", gridAndtime)
            grid_trajectory_index.append((gridAndtime[0].index, gridAndtime[1]))
        grid_trajectories_index.append(grid_trajectory_index)
    # print("grid_trajectories_index:", grid_trajectories_index)


"--------------------------------------------------------------------------------------------------------------"
# dataset_list = ['LDPTP_oldenburg', 'LDPTP_NYC', 'LDPTP_TKY', 'LDPTP_porto', 'LDPTP_Geolife']
dataset_list = ['LDPTP_oldenburg']
epsilon_list = [1]
time_num_list = [2]
grid_num_list = [4]
mu_list = [0.01]
loop = 2

# dataset_list = ['LDPTP_oldenburg']
# epsilon_list = [1]
# grid_num_list = [6]
# mu_list = [0.1]
# loop = 10


for datas in dataset_list:
    for epsilon in epsilon_list:
        for grid_num in grid_num_list:
            for mu in mu_list:
                for time_num in time_num_list:
                    # 读取配置文件
                    config = configparser.ConfigParser()
                    config.read('param_config.ini')

                    # 修改参数值
                    config.set('DEFAULT', 'dataset', datas)
                    config.set('DEFAULT', 'epsilon', str(epsilon))
                    config.set('DEFAULT', 'grid_num', str(grid_num))
                    config.set('DEFAULT', 'mu', str(mu))
                    config.set('DEFAULT', 'time_num', str(time_num))

                    # 保存更新后的配置文件
                    with open('param_config.ini', 'w') as configfile:
                        config.write(configfile)

                    args = get_args()
                    logger.info(
                        f'**** Get parameters：{args.dataset}--{args.epsilon}--{args.grid_num}--{args.mu}--{args.time_num} ****')
                    # print("dataset,epsilon,grid_num,mu:", args.dataset, args.epsilon, args.grid_num, args.mu)

                    Density_Error_list = list()
                    Hotspot_Query_Error_list = list()
                    Point_Query_AvRE_list = list()
                    Kendall_tau_list = list()
                    Trip_error_list = list()
                    Diameter_error_list = list()
                    Length_error_list = list()
                    Pattern_F1_error_list = list()
                    Pattern_support_error_list = list()
                    for l in range(loop):

                        logger.info(f'Reading {args.dataset} dataset...')
                        if args.dataset == 'LDPTP_oldenburg':
                            db = dataset.read_brinkhoff(args.dataset)
                        elif args.dataset == 'LDPTP_NYC':
                            db = dataset.read_brinkhoff(args.dataset)
                        elif args.dataset == 'LDPTP_TKY':
                            db = dataset.read_brinkhoff(args.dataset)
                        elif args.dataset == 'LDPTP_porto':
                            db = dataset.read_brinkhoff(args.dataset)
                        elif args.dataset == 'LDPTP_Geolife':
                            db = dataset.read_brinkhoff(args.dataset)
                        else:
                            logger.info(f'Invalid dataset: {args.dataset}')
                            db = None
                            exit()

                        random.shuffle(db)
                        # print(db)
                        stats = dataset.dataset_stats(db, f'../data/{args.dataset}_stats.json')
                        g_num = 1 / (2 * args.d) * math.sqrt((np.e ** args.epsilon - 1) / (2 * args.d) * math.sqrt(
                            np.log(2) / (2 * stats['points_num'])) + 2)
                        print("g_num:", g_num)

                        logger.info('---- 1. Convert raw trajectories to grids start ----')
                        """-----------1. 地理空间划分-------------"""
                        grid_map = GridMap(args.grid_num,
                                           stats['min_x'],
                                           stats['min_y'],
                                           stats['max_x'],
                                           stats['max_y'])

                        grid_trajectories = convert_raw_to_grid(db)
                        logger.info('**** 1. Convert raw trajectories to grids ****')
                        # print("--------1. Geospatial division completed----------")

                        # print("grid_trajectories:", grid_trajectories)

                        print("num_grid_trajectory:", len(grid_trajectories))

                        grid_trajectories_length = [len(sublist) for sublist in grid_trajectories]
                        print("len_grid_trajectory:", grid_trajectories_length)

                        # print_grid_trajectories_index(grid_trajectories)

                        if args.re_syn:
                            # 时间划分
                            time_parts = divide_day_into_parts(args.time_num)

                            """-----------2. 获取过渡状态候选区域-------------"""
                            grr_server = ldp.GRRServer(args.epsilon, grid_map)
                            grr_client = ldp.GRRClient(args.epsilon, grid_map, time_parts)

                            logger.info('---- 2. candidate regions start ----')

                            estimate_candidate_region(grid_trajectories, grr_client, grr_server, time_parts)
                            # print("grr_server.get_candidate_region():", grr_server.get_candidate_region())
                            logger.info('**** 2. Get candidate regions ****')
                            # print("--------2. Obtaining the transition state candidate area is completed----------")

                            # print("2. candidate_regions:", grr_server.get_candidate_region())
                            # print("grid_trajectories:", grid_trajectories)
                            # print("perturbed_grid_db_1:", grr_server.get_perturbed_grid_db_1())

                            """-----------3. 获取噪声过渡状态-------------"""
                            em_server = ldp.EMServer(args.epsilon, grr_server.get_candidate_regions())
                            em_client = ldp.EMClient(args.epsilon, grr_server.get_candidate_regions(), time_parts)
                            logger.info('---- 3 Get perturbed_transition start ----')
                            for t in grid_trajectories:
                                perturbed_t = list()
                                for i in range(len(t) - 1):
                                    cur_point_gridAndtime = t[i]
                                    next_point_gridAndtime = t[i + 1]
                                    "-----------3.1 获取过渡状态涉及的区域（索引）-------------"
                                    candidate_region_index = em_client.find_candidate_region_index(
                                        cur_point_gridAndtime,
                                        next_point_gridAndtime)
                                    # candidate_region_index (i,j)
                                    # print("candidate_region_index:", candidate_region_index)
                                    "-----------3.2 实例化所有可能得过渡状态，构建过渡状态候选空间-------------"
                                    candidate_transitions = em_client.generate_candidate_transition(
                                        candidate_region_index)
                                    # print("3.2 candidate_transitions:", candidate_transitions)

                                    "-----------3.3 利用EM输出噪声过渡状态-------------"
                                    perturbed_transition = em_client.perturb(candidate_transitions,
                                                                             cur_point_gridAndtime,
                                                                             next_point_gridAndtime)
                                    # print("perturbed_transition:", perturbed_transition)
                                    "-----------3.4 服务器 收集噪声过渡状态-------------"
                                    perturbed_t.append(perturbed_transition)
                                em_server.aggregate(perturbed_t)
                                "-----------3.5 服务器 收集扰动的起始/结束点-------------"
                                perturbed_start2end = em_client.perturb(candidate_transitions, t[0], t[-1])
                                em_server.aggregate_start2end(perturbed_start2end)

                            logger.info('**** 3 Get perturbed_transition end ****')
                            # print("--------3. Get noise transition state----------")

                            # print("candidate_transitions:", em_client.get_list_candidate_transitions())
                            # print("perturbed_transition:", em_server.get_perturbed_transitions_list())
                            lengths = [len(sublist) for sublist in em_server.get_perturbed_transitions_list()]
                            print("len_perturbed_transitions:", lengths)
                            print("num_perturbed_start2end:", len(em_server.get_perturbed_start2end_list()))

                            """-----------4. 过渡状态校正-------------"""
                            logger.info('---- 4. transition_correction start ----')
                            em_server.transition_correction(grr_server.get_perturbed_grid_db())

                            logger.info('**** 4. Get transition_correction ****')
                            # print("--------4. Transition state correction----------")

                            # print("corrected_transitions_list:", em_server.get_corrected_transitions_list())
                            lengths = [len(sublist) for sublist in em_server.get_corrected_transitions_list()]
                            print("len_corrected_transitions:", lengths)
                            print("num_corrected_start2end_list:", len(em_server.get_corrected_start2end()))
                            # print("corrected_start2end_list:", em_server.get_corrected_start2end())

                            """-----------5. 马尔科夫建模-------------"""
                            logger.info('---- 5. modeling_markov start ----')

                            one_level_mat = server_modeling_markov(em_server.get_corrected_transitions_list())
                            # start2end_dict: {31: [((1, 3), (1, 3)), ((2, 2), (3, 2)), ((2, 2), (2, 2))
                            # start2end_pro{31: [0.2, 0.2, 0.2, 0.2, 0.2], 15: [0.6666666666666666, 0.3333333333333333]
                            start2end_dict, start2end_pro = server_start2end_probability(
                                em_server.get_corrected_start2end(),
                                grid_trajectories_length)
                            logger.info('**** 5. modeling_markov end ****')

                            logger.info('---- 6. Synthesizing start ----')
                            synthetic_database = generate_synthetic_database(grid_trajectories_length,
                                                                             one_level_mat, start2end_dict,
                                                                             start2end_pro,
                                                                             len(db))

                            synthetic_trajectories = convert_grid_to_raw(synthetic_database)

                            logger.info('**** 6. Synthesizing end ****')

                            # 合成轨迹的文件路径
                            synthetic_trajectory_file_path = f'../data/{args.dataset}/syn_{args.dataset}_eps_{args.epsilon}_max_{args.max_len}_grid_{args.grid_num}.pkl'
                            # 检查目录是否存在，不存在则创建
                            os.makedirs(os.path.dirname(synthetic_trajectory_file_path), exist_ok=True)

                            with open(synthetic_trajectory_file_path, 'wb') as f:
                                pickle.dump(synthetic_trajectories, f)

                            synthetic_grid_trajectories = synthetic_database
                            # print("synthetic_grid_trajectories:", synthetic_grid_trajectories)
                        else:
                            try:
                                logger.info('Reading saved synthetic database...')
                                with open(
                                        f'../data/{args.dataset}/syn_{args.dataset}_eps_{args.epsilon}_max_{args.max_len}_grid_{args.grid_num}.pkl',
                                        'rb') as f:
                                    synthetic_trajectories = pickle.load(f)
                                synthetic_grid_trajectories = convert_raw_to_grid(synthetic_trajectories)
                            except FileNotFoundError:
                                logger.info('Synthesized file not found! Use --re_syn')
                                exit()

                        orig_trajectories = db
                        # orig_grid_trajectories = grid_trajectories
                        orig_grid_trajectories = [[gridAndtime[0] for gridAndtime in t] for t in grid_trajectories]
                        # print("orig_grid_trajectories:", orig_grid_trajectories)
                        orig_sampled_trajectories = convert_grid_to_raw(orig_grid_trajectories)
                        # print("orig_sampled_trajectories:", orig_sampled_trajectories)

                        # ============================ EXPERIMENTS =========================== #
                        np.random.seed(2022)
                        random.seed(2022)
                        logger.info('Experiment: Density Error...')
                        orig_density = get_real_density(orig_grid_trajectories)
                        syn_density = get_real_density(synthetic_grid_trajectories)
                        orig_density /= np.sum(orig_density)
                        syn_density /= np.sum(syn_density)
                        density_error = utils.jensen_shannon_distance(orig_density, syn_density)
                        logger.info(f'Density Error: {density_error}')

                        logger.info('Experiment: Hotspot Query Error...')
                        hotspot_ndcg = experiment.calculate_hotspot_ndcg(orig_density, syn_density)
                        logger.info(f'Hotspot Query Error: {1 - hotspot_ndcg}')
                        # Query AvRE
                        logger.info('Experiment: Query AvRE...')

                        queries = [SquareQuery(grid_map.min_x, grid_map.min_y, grid_map.max_x, grid_map.max_y,
                                               size_factor=args.size_factor) for
                                   _ in range(args.query_num)]

                        query_error = experiment.calculate_point_query(orig_sampled_trajectories,
                                                                       synthetic_trajectories,
                                                                       queries)
                        logger.info(f'Point Query AvRE: {query_error}')

                        # Location coverage Kendall-tau
                        logger.info('Experiment: Kendall-tau...')
                        kendall_tau = experiment.calculate_coverage_kendall_tau(orig_grid_trajectories,
                                                                                synthetic_grid_trajectories,
                                                                                grid_map)
                        logger.info(f'Kendall_tau:{kendall_tau}')

                        # Trip error
                        logger.info('Experiment: Trip error...')
                        orig_trip_dist, _, _ = get_start_end_dist(orig_grid_trajectories)
                        syn_trip_dist, _, _ = get_start_end_dist(synthetic_grid_trajectories)

                        orig_trip_dist = np.asarray(orig_trip_dist) / np.sum(orig_trip_dist)
                        syn_trip_dist = np.asarray(syn_trip_dist) / np.sum(syn_trip_dist)
                        trip_error = utils.jensen_shannon_distance(orig_trip_dist, syn_trip_dist)
                        logger.info(f'Trip error: {trip_error}')

                        # Diameter error
                        logger.info('Experiment: Diameter error...')
                        diameter_error = experiment.calculate_diameter_error(orig_trajectories, synthetic_trajectories,
                                                                             multi=args.multiprocessing)
                        logger.info(f'Diameter error: {diameter_error}')

                        # Length error
                        logger.info('Experiment: Length error...')
                        length_error = experiment.calculate_length_error(orig_trajectories, synthetic_trajectories)
                        logger.info(f'Length error: {length_error}')

                        # Pattern mining errors
                        logger.info('Experiment: Pattern mining errors...')
                        orig_pattern = experiment.mine_patterns(orig_grid_trajectories)
                        syn_pattern = experiment.mine_patterns(synthetic_grid_trajectories)

                        pattern_f1_error = experiment.calculate_pattern_f1_error(orig_pattern, syn_pattern)
                        pattern_support_error = experiment.calculate_pattern_support(orig_pattern, syn_pattern)

                        logger.info(f'Pattern F1 error: {pattern_f1_error}')
                        logger.info(f'Pattern support error: {pattern_support_error}')

                        Density_Error_list.append(density_error)
                        Hotspot_Query_Error_list.append(1 - hotspot_ndcg)
                        Point_Query_AvRE_list.append(query_error)
                        Kendall_tau_list.append(kendall_tau)
                        Trip_error_list.append(trip_error)
                        Diameter_error_list.append(diameter_error)
                        Length_error_list.append(length_error)
                        Pattern_F1_error_list.append(pattern_f1_error)
                        Pattern_support_error_list.append(pattern_support_error)

                    # 计算均值
                    Density_Error_mean = sum(Density_Error_list) / loop
                    Hotspot_Query_Error_mean = sum(Hotspot_Query_Error_list) / loop
                    Point_Query_AvRE_mean = sum(Point_Query_AvRE_list) / loop
                    Kendall_tau_mean = sum(Kendall_tau_list) / loop
                    Trip_error_mean = sum(Trip_error_list) / loop
                    Diameter_error_mean = sum(Diameter_error_list) / loop
                    Length_error_mean = sum(Length_error_list) / loop
                    Pattern_F1_error_mean = sum(Pattern_F1_error_list) / loop
                    Pattern_support_error_mean = sum(Pattern_support_error_list) / loop

                    # 写入文件
                    output_path = f'../data/output/'
                    output_mean = args.dataset + '_mean.txt'
                    with open(os.path.join(output_path, output_mean), 'a', encoding='utf-8') as f:
                        f.write(
                            f'dataset, epsilon, grid_num, mu, time_num: {args.dataset, args.epsilon, args.grid_num, args.mu, args.time_num}\n')
                        f.write(f'Density_Error_list 均值: {Density_Error_mean}\n')
                        f.write(f'Hotspot_Query_Error_list 均值: {Hotspot_Query_Error_mean}\n')
                        f.write(f'Point_Query_AvRE_list 均值: {Point_Query_AvRE_mean}\n')
                        f.write(f'Kendall_tau_list 均值: {Kendall_tau_mean}\n')
                        f.write(f'Trip_error_list 均值: {Trip_error_mean}\n')
                        f.write(f'Diameter_error_list 均值: {Diameter_error_mean}\n')
                        f.write(f'Length_error_list 均值: {Length_error_mean}\n')
                        f.write(f'Pattern_F1_error_list 均值: {Pattern_F1_error_mean}\n')
                        f.write(f'Pattern_support_error_list 均值: {Pattern_support_error_mean}\n')
                        f.write(f'----------------------------------\n')
                        f.flush()  # 立即写入到文件中

                        # 写入文件
                        output_path = f'../data/output/'
                        output_max = args.dataset + '_min.txt'
                        with open(os.path.join(output_path, output_max), 'a', encoding='utf-8') as f:
                            f.write(
                                f'dataset, epsilon, grid_num, mu, time_num: {args.dataset, args.epsilon, args.grid_num, args.mu, args.time_num}\n')
                            f.write(f'Density_Error_list 最小值: {min(Density_Error_list)}\n')
                            f.write(f'Hotspot_Query_Error_list 最小值: {min(Hotspot_Query_Error_list)}\n')
                            f.write(f'Point_Query_AvRE_list 最小值: {min(Point_Query_AvRE_list)}\n')
                            f.write(f'Kendall_tau_list 最小值: {min(Kendall_tau_list)}\n')
                            f.write(f'Trip_error_list 最小值: {min(Trip_error_list)}\n')
                            f.write(f'Diameter_error_list 最小值: {min(Diameter_error_list)}\n')
                            f.write(f'Length_error_list 最小值: {min(Length_error_list)}\n')
                            f.write(f'Pattern_F1_error_list 最小值: {min(Pattern_F1_error_list)}\n')
                            f.write(f'Pattern_support_error_list 最小值: {min(Pattern_support_error_list)}\n')
                            f.write(f'----------------------------------\n')
                            f.flush()  # 立即写入到文件中
