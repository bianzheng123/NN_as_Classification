from procedure_nn_classification.init_0 import multiple_base_partition
import numpy as np
import sklearn.cluster as cls
from procedure_nn_classification.dataset_partition_1.kmeans import independent_kmeans


class IndependentKMeans(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(IndependentKMeans, self).__init__(config)
        self.specific_type = config['specific_type']
        self.obj_id = '%s_%s' % (self.type, self.specific_type)
        # program_train_para_dir, n_cluster, n_instance, entity_number, models

    def get_model(self, config):
        return independent_kmeans.KMeans(config)

    def _preprocess(self, base):
        pass
