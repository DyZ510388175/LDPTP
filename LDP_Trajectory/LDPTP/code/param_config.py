# update_config.py
import configparser

# 读取配置文件
config = configparser.ConfigParser()
config.read('param_config.ini')

# 设置默认参数值
config.set('DEFAULT', 'time_num', '4') # Number of time periods
config.set('DEFAULT', 'd', '10') # maximum distance
config.set('DEFAULT', 'mu', '0.1') # scaling parameter
config.set('DEFAULT', 'w1', '0.1') # weight parameter

config.set('DEFAULT', 'epsilon', '1.0') # Privacy budget
config.set('DEFAULT', 'grid_num', '6') # Number of grids is n x n
config.set('DEFAULT', 'query_num', '200') # Number of experiment queries

# LDPTP_oldenburg LDPTP_NYC LDPTP_TKY LDPTP_porto LDPTP_Geolife
config.set('DEFAULT', 'dataset', 'LDPTP_oldenburg')
config.set('DEFAULT', 're_syn', 'True') # Synthesizing or use existing file
config.set('DEFAULT', 'max_len', '0.9') # Quantile of estimated max length
config.set('DEFAULT', 'size_factor', '9.0') # Quantile of estimated max length
config.set('DEFAULT', 'max_len', '0.9') # Quantile of estimated max length

# 保存更新后的配置文件
with open('param_config.ini', 'w') as configfile:
    config.write(configfile)
