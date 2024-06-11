from typing import List, Tuple
import numpy as np
import json
import pickle
import trajectory


def read_brinkhoff(dataset='brinkhoff'):
    """
    Brinkhoff dataset:
    #n:
    >0: x1,y1;x2,y2;...:
    """
    db = []
    file_name = f'E:/experimental_dataset/trajectory_dataset/{dataset}.dat'
    # file_name = f'./LDPTrace/data/{dataset}.dat'
    with open(file_name, 'r') as f:
        row = f.readline()
        while row:
            if row[0] == '#':
                row = f.readline()
                continue
            if not row[0] == '>':
                print(row)
                exit()
            # Skip '>0:' and ';\n' in the end
            row = row[3:-2].split(';')  # row: ['x1,y1', 'x2,y2', ...]

            t = [x.split(',') for x in row]  # t: [['x1','y1'], ['x2','y2'], ...]
            # print("t:",t)

            # t = [(eval(x[0]), eval(x[1]), x[2]) for x in t]  # t: [(x1,y1), (x2,y2), ...]
            t = [(eval(x[0]), eval(x[1]), time_to_minutes(x[2])) for x in t]  # t: [(x1,y1), (x2,y2), ...]

            db.append(t)
            row = f.readline()

    return db

# 将时间字符串转换为分钟数
def time_to_minutes(time_str):
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = int(parts[2])
    total_minutes = hours * 60 + minutes + seconds / 60
    return total_minutes

def dataset_stats(db: List[List[Tuple[float, float, str]]], db_name: str):
    lengths = np.asarray([len(t) for t in db])

    xs = [[p[0] for p in t] for t in db]
    ys = [[p[1] for p in t] for t in db]

    min_xs = [min(x) for x in xs]
    min_ys = [min(y) for y in ys]
    max_xs = [max(x) for x in xs]
    max_ys = [max(y) for y in ys]

    stats = {
        # 轨迹数量，长度
        'num': len(db),
        'min_len': int(min(lengths)),
        'max_len': int(max(lengths)),
        'mean_len': float(np.mean(lengths)),
        'points_num': len(db) * int(np.mean(lengths)),
        #坐标
        'min_x': min(min_xs),
        'min_y': min(min_ys),
        'max_x': max(max_xs),
        'max_y': max(max_ys)
    }

    print(stats)

    with open(db_name, 'w') as f:
        json.dump(stats, f)

    return stats