import numpy as np
import os
import copy

'''
子类需要重写get_model方法, 返回模型
还需要重写preprocess方法
'''


class MultipleBasePartition:
    def __init__(self, config):
        self.program_train_para_dir = config['program_train_para_dir']
        self.type = config['type']
        self.n_instance = config['n_instance']
        self.n_cluster = config['n_cluster']
        self.kahip_dir = config['kahip_dir']
        self.entity_number = config['entity_number']
        self.model_l = []
        # 用于识别的标识
        self.obj_id = None
        for i in range(self.n_instance):
            tmp_config = copy.deepcopy(config['dataset_partition'])
            tmp_config['type'] = self.type
            tmp_config['classifier_number'] = i + 1
            tmp_config['entity_number'] = self.entity_number
            tmp_config['n_cluster'] = self.n_cluster
            tmp_config['kahip_dir'] = self.kahip_dir
            tmp_config['save_dir'] = '%s/Classifier_%d_%d' % (
                self.program_train_para_dir, self.entity_number, tmp_config['classifier_number'])
            os.system('mkdir %s' % tmp_config['save_dir'])
            tmp_model = self.get_model(tmp_config)
            self.model_l.append(tmp_model)

    # 直接返回模型的实例
    def preprocess(self, base):
        print('start preprocessing %s_%d' % (self.obj_id, self.entity_number))
        self._preprocess(base)
        print('finish preprocessing %s_%d' % (self.obj_id, self.entity_number))
        return self.model_l
