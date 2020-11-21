import numpy as np
import os
import torch


class BaseDataNode:
    def __init__(self, config):
        self.type = config['type']
        self.output_type = config['output']
        # 保存该模型参数的地址
        self.save_dir = '%s/prepare_train_sample' % config['save_dir']
        os.system('mkdir %s' % self.save_dir)
        self.classifier_number = config['classifier_number']

        self.n_cluster = config['n_cluster']
        self.label_k = config['label_k']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.train_split = config['train_split']

        self.trainloader = None
        self.valloader = None

    def prepare(self, base, partition_info):
        pass

    def save(self):
        save_trainloader_dir = '%s/trainloader.pth' % self.save_dir
        torch.save(self.trainloader, save_trainloader_dir)
        save_valloader_dir = '%s/valloader.pth' % self.save_dir
        torch.save(self.valloader, save_valloader_dir)

    def __str__(self):
        return '%s_%d, output_type: %s, save_dir: %s' % (
            self.type, self.classifier_number, self.output_type, self.save_dir)
