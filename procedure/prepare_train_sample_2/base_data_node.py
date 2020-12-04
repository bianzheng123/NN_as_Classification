import numpy as np
import os
import torch
import time


class BaseDataNode:
    def __init__(self, config):
        self.type = config['type']
        self.output_type = config['output']
        # 保存该模型参数的地址
        self.save_dir = '%s/prepare_train_sample' % config['save_dir']
        self.entity_number = config['entity_number']
        self.classifier_number = config['classifier_number']
        self.obj_id = "%s_%d_%d" % (self.type, self.entity_number, self.classifier_number)

        self.n_cluster = config['n_cluster']
        self.label_k = config['label_k']
        self.batch_size = config['batch_size']
        self.shuffle = config['shuffle']
        self.train_split = config['train_split']

        self.trainloader = None
        self.valloader = None

    def prepare(self, base, base_base_gnd, partition_info):
        start_time = time.time()
        print('start prepare data %s %s' % (self.obj_id, self.output_type))
        self._prepare(base, base_base_gnd, partition_info)
        print('finish prepare_data %s %s' % (self.obj_id, self.output_type))
        end_time = time.time()
        intermediate_config = {
            'time': end_time - start_time
        }
        return (self.trainloader, self.valloader), intermediate_config

    '''
    Input:
    -base
    -partition_info
    Output:
    -已经打包好的训练集以及测试集
    将打包好的数据集放到self.trainloader和self.valloader中
    '''

    def _prepare(self, base, partition_info):
        pass

    def save(self):
        os.system('mkdir %s' % self.save_dir)
        save_trainloader_dir = '%s/trainloader.pth' % self.save_dir
        torch.save(self.trainloader, save_trainloader_dir)
        save_valloader_dir = '%s/valloader.pth' % self.save_dir
        torch.save(self.valloader, save_valloader_dir)

    def __str__(self):
        return '%s, output_type: %s, save_dir: %s' % (
            self.obj_id, self.output_type, self.save_dir)
