from procedure_counting_index.dataset_partition_1 import base_partition
import numpy as np
import sklearn.cluster as cls


class LSH(base_partition.BasePartition):

    def __init__(self, config):
        super(LSH, self).__init__(config)
        self.r = config['r']
        self.a_sigma = config['a_sigma']
        self.a_miu = config['a_miu']
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base):
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

    def _predict(self, query):
        query_norm = query / self.norm_div
        proj = np.dot(query_norm, self.a)
        predict = np.floor((proj + self.b) / self.r) % self.n_cluster
        predict = predict.astype(np.int)
        return predict

    def __str__(self):
        string = super(KMeans, self).__str__()
        string = '%s r: %d, a_sigma: %d, a_miu: %d' % (string, self.r, self.a_sigma, self.a_miu)
        return string
