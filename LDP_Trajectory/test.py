# common_args.py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--grid_num', type=int, default=6, help='Number of grids is n x n')
args = parser.parse_args()

# A.py
from common_args import args

# 修改参数值
args.grid_num = 4
