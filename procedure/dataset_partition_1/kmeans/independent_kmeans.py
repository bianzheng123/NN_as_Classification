from procedure.dataset_partition_1 import base_partition
import numpy as np
import sklearn.cluster as cls


class KMeans(base_partition.BasePartition):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        self.max_iter = config['max_iter']
        self.model = cls.KMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=self.max_iter)
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def partition(self, base):
        self.model.fit(base)
        self.labels = self.model.labels_
        # 确定一个self.labels即可
        self.get_labels(self.labels)
        super(KMeans, self).get_model_info()
        return (self.labels, self.label_map), self.model_info

    def __str__(self):
        string = super(KMeans, self).__str__()
        string = '%s max_iter: %s' % (string, self.max_iter)
        return string
