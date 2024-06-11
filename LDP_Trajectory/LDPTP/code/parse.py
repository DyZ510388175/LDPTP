import argparse
import configparser

# # 读取配置文件
# config = configparser.ConfigParser()
# config.read('config.ini')
#
# parser = argparse.ArgumentParser()
#
# parser.add_argument('--time_num', type=int, default=config.getint('DEFAULT', 'time_num'),
#                     help='Number of time periods')
# parser.add_argument('--d', type=int, default=config.getint('DEFAULT', 'd'),
#                     help='maximum distance')
# parser.add_argument('--mu', type=float, default=config.getfloat('DEFAULT', 'mu'),
#                     help='scaling parameter')
# parser.add_argument('--w1', type=float, default=config.getfloat('DEFAULT', 'w1'),
#                     help='weight parameter')
#
# parser.add_argument('--epsilon', type=float, default=config.getfloat('DEFAULT', 'epsilon'),
#                     help='Privacy budget')
# parser.add_argument('--grid_num', type=int, default=config.getint('DEFAULT', 'grid_num'),
#                     help='Number of grids is n x n')
# parser.add_argument('--query_num', type=int, default=config.getint('DEFAULT', 'query_num'),
#                     help='Number of experiment queries')
#
# # LDPTP_oldenburg LDPTP_NYC LDPTP_TKY LDPTP_porto LDPTP_Geolife
# parser.add_argument('--dataset', type=str, default=config.get('DEFAULT', 'dataset'))
# parser.add_argument('--re_syn', action='store_true', default=config.get('DEFAULT', 're_syn'),
#                     help='Synthesizing or use existing file')
# parser.add_argument('--max_len', type=float, default=config.getfloat('DEFAULT', 'max_len'),
#                     help='Quantile of estimated max length')
# parser.add_argument('--size_factor', type=float, default=config.getfloat('DEFAULT', 'size_factor'),
#                     help='Quantile of estimated max length')
# parser.add_argument('--multiprocessing', action='store_true')
#
#
# args = parser.parse_args()

def get_args():
    #读取配置文件
    config = configparser.ConfigParser()
    config.read('param_config.ini')

    parser = argparse.ArgumentParser()

    parser.add_argument('--time_num', type=int, default=config.getint('DEFAULT', 'time_num'),
                        help='Number of time periods')
    parser.add_argument('--d', type=int, default=config.getint('DEFAULT', 'd'),
                        help='maximum distance')
    parser.add_argument('--mu', type=float, default=config.getfloat('DEFAULT', 'mu'),
                        help='scaling parameter')
    parser.add_argument('--w1', type=float, default=config.getfloat('DEFAULT', 'w1'),
                        help='weight parameter')

    parser.add_argument('--epsilon', type=float, default=config.getfloat('DEFAULT', 'epsilon'),
                        help='Privacy budget')
    parser.add_argument('--grid_num', type=int, default=config.getint('DEFAULT', 'grid_num'),
                        help='Number of grids is n x n')
    parser.add_argument('--query_num', type=int, default=config.getint('DEFAULT', 'query_num'),
                        help='Number of experiment queries')

    # LDPTP_oldenburg LDPTP_NYC LDPTP_TKY LDPTP_porto LDPTP_Geolife
    parser.add_argument('--dataset', type=str, default=config.get('DEFAULT', 'dataset'))
    parser.add_argument('--re_syn', action='store_true', default=config.get('DEFAULT', 're_syn'),
                        help='Synthesizing or use existing file')
    parser.add_argument('--max_len', type=float, default=config.getfloat('DEFAULT', 'max_len'),
                        help='Quantile of estimated max length')
    parser.add_argument('--size_factor', type=float, default=config.getfloat('DEFAULT', 'size_factor'),
                        help='Quantile of estimated max length')
    parser.add_argument('--multiprocessing', action='store_true')


    # args = parser.parse_args()
    return parser.parse_args()
