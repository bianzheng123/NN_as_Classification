from procedure.dataset_partition_1 import multiple_base_partition
from procedure.dataset_partition_1.kmeans import kmeans
import numpy as np
import sklearn.cluster as cls


class MultipleKMeans(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(MultipleKMeans, self).__init__(config)
        # 用于构建m个kmeans的质心们, m * k * d
        self.centroid_l_l = None
        self.max_iter = config['max_iter']
        self.model = cls.KMeans(n_clusters=self.n_cluster * self.n_instance, init='k-means++', max_iter=self.max_iter)

    def get_model(self, config):
        return kmeans.KMeans(config)

    def partition(self, base):
        print('start training mkmeans_%d' % self.entity_number)
        # 得到m个kmeans的质心
        self.model.fit(base)
        # mk * d
        centroids = self.model.cluster_centers_
        centroid_sort_idx = self.random_projection(centroids)
        # random_projection将数组进行排序, 使其符合k组, 每组m个的形式
        centroid_sort_idx = centroid_sort_idx.reshape(self.n_cluster, -1)

        # 在这k组中每组各提取一个向量, 一个模型的质心
        centroid_l_l = []
        for i in range(self.n_instance):
            model_centroid_l = []
            for j in range(self.n_cluster):
                idx = centroid_sort_idx[j][i]
                model_centroid_l.append(centroids[idx])
            centroid_l_l.append(model_centroid_l)
        self.centroid_l_l = np.array(centroid_l_l)

        # 将这些质心分配给各个模型实例
        for i in range(self.n_instance):
            self.models[i].get_centroid(self.centroid_l_l[i])
        # 得到质心后, 计算与base的距离, 得到label
        partition_info_l = super(MultipleKMeans, self).partition(base)
        print('finish training mkmeans_%d' % self.entity_number)
        return partition_info_l

    def random_projection(self, centroid_l):
        res_idx = np.arange(self.n_cluster * self.n_instance)
        self.divide_and_conquer(0, self.n_cluster, centroid_l, 0, len(centroid_l), res_idx)
        return res_idx

    @staticmethod
    def divide_and_conquer(depth, k, centroid_l, start, end, res_idx):
        if 2 ** depth == k:
            return
        random_vector = np.random.normal(centroid_l.shape[1])
        random_l = None
        for i in range(len(centroid_l)):
            if i == 0:
                random_num = np.dot(random_vector, centroid_l[i])
                random_l = np.array([random_num])
            random_num = np.dot(random_vector, centroid_l[i])
            to_append = np.array([random_num])
            random_l = np.append(random_l, to_append)
        # random_l是质心和高斯分布点乘后的结果
        depth += 1
        sort_indices = np.argsort(random_l[start:end]) + start
        mid = int((start + end - 1) / 2)
        res_idx[start:end] = sort_indices
        MultipleKMeans.divide_and_conquer(depth, k, centroid_l, start, mid, res_idx)
        MultipleKMeans.divide_and_conquer(depth, k, centroid_l, mid, end, res_idx)
