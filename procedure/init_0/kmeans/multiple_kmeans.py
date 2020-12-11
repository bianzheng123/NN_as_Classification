from procedure.init_0.kmeans import base_multiple_kmeans
import numpy as np
import sklearn.cluster as cls


class MultipleKMeans(base_multiple_kmeans.BaseMultipleKMeans):

    def __init__(self, config):
        super(MultipleKMeans, self).__init__(config)
        # 用于构建m个kmeans的质心们, m * k * d
        self.model = cls.KMeans(n_clusters=self.n_cluster * self.n_instance, init='k-means++',
                                max_iter=self.max_iter)
