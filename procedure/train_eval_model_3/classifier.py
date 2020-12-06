import numpy as np
import os
import time
import torch


class Classifier:
    def __init__(self, config):
        self.type = config['type']
        # 保存该模型参数的地址
        self.save_dir = '%s/train_model' % config['save_dir']
        self.entity_number = config['entity_number']
        self.classifier_number = config['classifier_number']
        self.obj_id = "%s_%d_%d" % (self.type, self.entity_number, self.classifier_number)
        self.n_cluster = config['n_cluster']
        # 就是score_table, query.shape[0] * base.shape[0]
        self.result = None
        # 存放中间结果参数
        self.intermediate_config = {}

    def train(self, base, trainset):
        start_time = time.time()
        print('start training %s' % self.obj_id)
        self._train(base, trainset)
        print('finish training %s' % self.obj_id)
        end_time = time.time()
        self.intermediate_config['training_time'] = end_time - start_time

    '''
    训练, 训练后的参数放到train_para中
    -base 数据集
    -trainset 包括了trainloader, valloader, 是trainset的一个实例
    '''

    def _train(self, base, trainset):
        pass

    def eval(self, query):
        start_time = time.time()
        print('start evaluate %s' % self.obj_id)
        self._eval(query)
        print('finish evaluate %s' % self.obj_id)
        end_time = time.time()
        self.intermediate_config['eval_time'] = end_time - start_time
        return self.result, self.intermediate_config

    '''
    query是二维数组, 批量处理
    '''

    def _eval(self, query):
        pass

    def save(self):
        os.system('mkdir %s' % self.save_dir)
        eval_res_dir = '%s/eval_res.txt' % self.save_dir
        np.savetxt(eval_res_dir, self.result, fmt='%.3f')

    def __str__(self):
        return '%s, save_dir: %s' % (
            self.obj_id, self.save_dir)
