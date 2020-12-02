from procedure.init_0 import multiple_base_partition
import numpy as np
import sklearn.cluster as cls
from procedure.dataset_partition_1.kmeans import independent_kmeans


class IndependentKMeans(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(IndependentKMeans, self).__init__(config)
        # program_train_para_dir, n_cluster, n_instance, entity_number, models

    def get_model(self, config):
        return independent_kmeans.KMeans(config)
