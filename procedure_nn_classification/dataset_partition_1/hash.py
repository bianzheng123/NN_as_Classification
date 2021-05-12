from procedure_nn_classification.dataset_partition_1 import base_partition
import numpy as np
import time
from util import dir_io


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
        if 'a_sigma' in config:
            self.a_sigma = config['a_sigma']
            print("a_sigma " + str(self.a_sigma))
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base, base_base_gnd, ins_intermediate):
        start_time = time.time()
        norm = np.linalg.norm(base, axis=1)
        # print(norm)
        self.norm_div = np.max(norm)
        # print(norm_div)
        base_normlize = base / self.norm_div
        self.a = np.random.normal(loc=self.a_miu, scale=self.a_sigma, size=base.shape[1])
        proj_result = np.dot(base_normlize, self.a)
        self.b = np.random.random() * self.r
        arr = np.floor((proj_result + self.b) / self.r) % self.n_cluster
        self.labels = arr.astype(np.int)
        partition_dir = '%s/partition.txt' % self.save_dir
        dir_io.save_array_txt(partition_dir, self.labels, '%d')
        end_time = time.time()
        self.intermediate['hashing_time'] = end_time - start_time
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels, self.distance_metric


class RandomProjection(base_partition.BasePartition):
    def __init__(self, config):
        super(RandomProjection, self).__init__(config)
        self.partition_depth = int(np.log2(self.n_cluster))
        self.rp_miu = 0
        self.rp_sigma = 100
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base, base_base_gnd, ins_intermediate):
        start_time = time.time()

        # use random projection to project the data to 2 part
        partition_idx = self.random_projection(base)

        n_part = 2 ** self.partition_depth
        labels = np.empty(len(base), dtype=np.int)

        start_idx = 0
        for i in range(n_part):
            end_idx = int(np.ceil(len(base) / n_part)) * (i + 1)
            labels[partition_idx[start_idx:end_idx]] = i
            start_idx = end_idx
        self.labels = labels

        partition_dir = '%s/partition.txt' % self.save_dir
        dir_io.save_array_txt(partition_dir, self.labels, '%d')
        end_time = time.time()
        self.intermediate['hashing_time'] = end_time - start_time
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels, self.distance_metric

    def random_projection(self, base):
        res_idx = np.arange(len(base))
        self.divide_and_conquer(0, base, 0, len(base), res_idx)
        return res_idx

    def divide_and_conquer(self, depth, data, start, end, res_idx):
        if depth == self.partition_depth:
            return
        # vector = np.random.rand(data.shape[1])
        # random_vector = vector / np.linalg.norm(vector)
        random_vector = np.random.normal(size=data.shape[1], scale=self.rp_sigma, loc=self.rp_miu)
        random_l = []
        for i in range(start, end):
            random_num = np.dot(random_vector, data[res_idx[i]])
            random_l.append(random_num)
        # random_l is the result of dot product of centroid and random vector(follow Gauss distribution)
        random_l = np.array(random_l)
        depth += 1
        sort_indices = np.argsort(random_l) + start

        mid = int((start + end) / 2 + 1)
        res_idx[start:end] = sort_indices
        self.divide_and_conquer(depth, data, start, mid, res_idx)
        self.divide_and_conquer(depth, data, mid, end, res_idx)
