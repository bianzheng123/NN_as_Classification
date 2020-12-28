import _init_paths
from procedure_nn_classification.train_eval_model_3 import train_eval_model
import torch
import numpy as np
import json
import math
from procedure_common import result_integrate

# gnd = np.load('/home/bianzheng/NN_as_Classification/data/sift_10/gnd.npy')
gnd = np.load('/home/bianzheng/NN_as_Classification/data/siftsmall_10/gnd.npy')
# score_table_knn = np.load(
#     '/home/bianzheng/NN_as_Classification/train_para/sift_256_nn_4_knn_parhip/total_score_table.npy')
score_table = np.load(
    '/home/bianzheng/NN_as_Classification/pytest/data/score_table/total_score_table_siftsmall.npy')
result_integrate_config = {
    'k': 10,
    'program_result_dir': '/home/bianzheng/NN_as_Classification/result/Pytest/test',
    # 'efSearch_l': [2 ** i for i in range(1 + int(math.log2(1000000)))],
    'efSearch_l': [2 ** i for i in range(1 + int(math.log2(10000)))],
}
result_integrate.integrate_single(score_table, gnd, result_integrate_config)
