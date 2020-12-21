from procedure_nn_classification.dataset_partition_1 import base_partition
import numpy as np
import sklearn.cluster as cls
import time


class KMeans(base_partition.BasePartition):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        self.max_iter = config['max_iter']
        self.model = cls.KMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=self.max_iter)
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base, obj):
        kmeans_start_time = time.time()
        self.model.fit(base)
        kmeans_end_time = time.time()
        self.intermediate['kmeans_time'] = kmeans_end_time - kmeans_start_time
        self.labels = self.model.labels_

    def __str__(self):
        string = super(KMeans, self).__str__()
        string = '%s max_iter: %s' % (string, self.max_iter)
        return string
