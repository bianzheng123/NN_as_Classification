from procedure_counting_index.init_0 import preprocess_base_partition
import numpy as np
import sklearn.cluster as cls
from procedure_counting_index.dataset_partition_1 import lsh


class PreprocessLSH(preprocess_base_partition.PreprocessBasePartition):

    def __init__(self, config):
        super(PreprocessLSH, self).__init__(config)
        self.obj_id = '%s' % self.type
        # program_train_para_dir, n_cluster, n_instance, entity_number, models

    def get_model(self, config):
        return lsh.LSH(config)

    def _preprocess(self, base):
        pass
