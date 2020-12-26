import _init_paths
from procedure_nn_classification.train_eval_model_3 import train_eval_model
import numpy as np
import random

cls_score_l = np.array([
    [[random.random(), random.random(), random.random(), random.random()], [random.random(), random.random(), random.random(), random.random()]],
    [[random.random(), random.random(), random.random(), random.random()], [random.random(), random.random(), random.random(), random.random()]]
])
label_m_l = [
    {
        0: [0, 1, 2],
        1: [3, 4, 5],
        2: [6, 7, 8],
        3: [9, 10, 11]
    },
    {
        0: [0, 11, 10],
        1: [2, 3, 4],
        2: [1, 6, 7],
        3: [5, 8, 9]
    }
]
config = {
    'n_item': 12,
    'program_train_para_dir': '/home/bianzheng/NN_as_Classification/pytest/data/score_table'
}

train_eval_model.integrate_save_score_table_parallel(cls_score_l, label_m_l, config, True)
train_eval_model.integrate_save_score_table_total(cls_score_l, label_m_l, config, True)
