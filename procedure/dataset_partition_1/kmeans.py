from procedure.dataset_partition_1 import base_partition
import numpy as np
import sklearn.cluster as cls


class KMeans(base_partition.BasePartition):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        # 该模型需要聚类的数量
        self.model = cls.KMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=30)
        # self.type, self.save_dir, self.classifier_number, self.label, self.n_cluster

    def partition(self, base):
        print('start training %s_%d' % (self.type, self.classifier_number))
        self.model.fit(base)
        self.get_labels(self.model.labels_)
        print('finish training %s_%d' % (self.type, self.classifier_number))
        return self.model.labels_, self.label
