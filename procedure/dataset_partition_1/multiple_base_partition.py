import numpy as np
import os
import copy


class MultipleBasePartition:
    def __init__(self, config):
        self.program_train_para_dir = config['program_train_para_dir']
        self.n_cluster = config['n_cluster']
        self.n_instance = config['n_instance']
        self.entity_number = config['entity_number']
        self.models = []
        for i in range(self.n_instance):
            tmp_config = copy.deepcopy(config)
            tmp_config['classifier_number'] = i + 1
            tmp_config['save_dir'] = '%s/Classifier_%d_%d' % (
                self.program_train_para_dir, self.entity_number, tmp_config['classifier_number'])
            os.system('mkdir %s' % tmp_config['save_dir'])
            tmp_model = self.get_model(tmp_config)
            self.models.append(tmp_model)

    def get_model(self, config):
        pass

    def partition(self, base):
        partition_info_l = []
        for model in self.models:
            info = model.partition(base)
            partition_info_l.append(info)
        return partition_info_l

    def save(self):
        for model in self.models:
            model.save()

    def __str__(self):
        res_str = ''
        for model in self.models:
            res_str += model.__str__() + '\n'
        return res_str
