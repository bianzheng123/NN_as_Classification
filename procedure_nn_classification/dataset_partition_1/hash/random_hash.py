from procedure_nn_classification.dataset_partition_1 import base_partition
import numpy as np
import time


class RandomHash(base_partition.BasePartition):

    def __init__(self, config):
        super(RandomHash, self).__init__(config)
        self.range = self.n_cluster
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels, self.distance_metric

    def _partition(self, base, base_base_gnd, ins_intermediate):
        start_time = time.time()
        # generate a random number in the range, then mod self.n_cluster as the label
        labels = np.random.randint(0, self.n_cluster, size=(len(base)))
        end_time = time.time()
        self.intermediate['kmeans_time'] = end_time - start_time
        self.labels = labels
