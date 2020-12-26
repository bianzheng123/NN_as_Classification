from procedure_nn_classification.train_eval_model_3 import train_eval_model
import torch
import numpy as np
import json
import math
from procedure_common import result_integrate

gnd = np.load('/home/bianzheng/NN_as_Classification/data/sift_10/gnd.npy')
score_table = np.load(
    '/home/bianzheng/NN_as_Classification/train_para/sift_256_nn_1_kmeans_multiple/total_score_table.npy')
print("load success")
result_integrate_config = {
    'k': 10,
    'program_result_dir': '/home/bianzheng/NN_as_Classification/result/sift_256_nn_1_kmeans_multiple',
    'efSearch_l': [2 ** i for i in range(1 + int(math.log2(1000000)))],
}
count_recall_intermediate = result_integrate.integrate_single(score_table, gnd, result_integrate_config)
