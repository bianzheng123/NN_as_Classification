import numpy as np
import os
import torch


class Classifier:
    def __init__(self, config):
        self.type = config['type']
        # 保存该模型参数的地址
        self.save_dir = '%s/train_model' % config['save_dir']
        os.system('mkdir %s' % self.save_dir)
        self.classifier_number = config['classifier_number']
        self.n_cluster = config['n_cluster']

        self.train_para = None

    def train(self, base, trainset):
        pass

    def eval(self, query):
        pass

    def save(self):
        train_para_dir = '%s/train_para.pth' % self.save_dir
        torch.save(self.train_para, train_para_dir)

    def __str__(self):
        return '%s_%d, save_dir: %s' % (
            self.type, self.classifier_number, self.save_dir)
