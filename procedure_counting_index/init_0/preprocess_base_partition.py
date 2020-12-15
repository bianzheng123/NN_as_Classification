import numpy as np
import os
import copy

'''
the son class should rewrite the funciton of get_model() and preprocess()
get_model() is to return the model of classifier
preprocess is to return the array of the model
'''


class PreprocessBasePartition:
    def __init__(self, config):
        self.program_train_para_dir = config['program_train_para_dir']
        self.type = config['type']
        self.n_instance = config['n_instance']
        self.n_cluster = config['n_cluster']
        self.entity_number = config['entity_number']
        self.model_l = []
        # for identification
        self.obj_id = None
        for i in range(self.n_instance):
            tmp_config = copy.deepcopy(config['dataset_partition'])
            tmp_config['type'] = self.type
            tmp_config['classifier_number'] = i + 1
            tmp_config['entity_number'] = self.entity_number
            tmp_config['n_cluster'] = self.n_cluster
            tmp_config['save_dir'] = '%s/Classifier_%d_%d' % (
                self.program_train_para_dir, self.entity_number, tmp_config['classifier_number'])
            os.system('mkdir %s' % tmp_config['save_dir'])
            tmp_model = self.get_model(tmp_config)
            self.model_l.append(tmp_model)

    # the son class should set the list of model to self.model_l
    def preprocess(self, base):
        print('start preprocessing %s_%d' % (self.obj_id, self.entity_number))
        self._preprocess(base)
        print('finish preprocessing %s_%d' % (self.obj_id, self.entity_number))
        return self.model_l
