import numpy as np
import time
import torch
from util import dir_io


class Classifier:
    def __init__(self, config):
        self.type = config['type']
        self.save_dir = '%s/train_model' % config['save_dir']
        self.classifier_number = config['classifier_number']
        self.obj_id = "%s_%d" % (self.type, self.classifier_number)
        self.n_cluster = config['n_cluster']
        # this is the score_table, it stores the score of each classifier for each query, its shape is query.shape[0] * base.shape[0]
        self.result = None
        # store the intermediate parameter such as loss and time consumed
        self.intermediate_config = {}

    def train(self, base, trainset):
        start_time = time.time()
        print('start training %s' % self.obj_id)
        self._train(base, trainset)
        print('finish training %s' % self.obj_id)
        end_time = time.time()
        self.intermediate_config['training_time'] = end_time - start_time

    '''
    -base dataset
    -trainset include trainloader, valloader, which is a instance of trainset
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
    query is a 2d array, this function enable batch process
    '''

    def _eval(self, query):
        pass

    def save(self):
        dir_io.mkdir(self.save_dir)
        eval_res_dir = '%s/eval_res.txt' % self.save_dir
        dir_io.save_array_txt(eval_res_dir, self.result, fmt='%.3f')

    def __str__(self):
        return '%s, save_dir: %s' % (
            self.obj_id, self.save_dir)
