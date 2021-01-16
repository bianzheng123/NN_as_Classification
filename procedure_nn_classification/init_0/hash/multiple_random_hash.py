from procedure_nn_classification.dataset_partition_1.hash import random_hash
from procedure_nn_classification.init_0 import multiple_base_partition
import copy


class MultipleRandomHash(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(MultipleRandomHash, self).__init__(config)
        if self.distance_metric != 'l2':
            raise Exception("not support distance metric")
        self.obj_id = 'random_hash'

    def get_model(self, config):
        return random_hash.RandomHash(config)

    def _preprocess(self, base):
        return {}
