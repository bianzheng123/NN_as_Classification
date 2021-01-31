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
        self.intermediate['hashing_time'] = end_time - start_time
        self.labels = labels


class LocalitySensitiveHash(base_partition.BasePartition):
    def __init__(self, config):
        super(LocalitySensitiveHash, self).__init__(config)
        self.r = 1
        self.a_sigma = 1
        self.a_miu = 0
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base, base_base_gnd, ins_intermediate):
        start_time = time.time()
        norm = np.linalg.norm(base, axis=1)
        # print(norm)
        self.norm_div = np.max(norm)
        # print(norm_div)
        base_normlize = base / self.norm_div
        self.a = np.random.normal(size=base.shape[1])
        proj_result = np.dot(base_normlize, self.a)
        self.b = np.random.random() * self.r
        arr = np.floor((proj_result + self.b) / self.r) % self.n_cluster
        self.labels = arr.astype(np.int)
        end_time = time.time()
        self.intermediate['hashing_time'] = end_time - start_time
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels, self.distance_metric
