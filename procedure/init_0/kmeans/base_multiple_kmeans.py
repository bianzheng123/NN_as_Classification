from procedure.init_0 import multiple_base_partition
from procedure.dataset_partition_1.kmeans import multiple_kmeans
import numpy as np
import sklearn.cluster as cls

'''
子类需要
1. 确定用于mk聚类的kmeans模型即self.model
'''


class BaseMultipleKMeans(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(BaseMultipleKMeans, self).__init__(config)
        # 用于构建m个kmeans的质心们, m * k * d
        self.specific_type = config['specific_type']
        self.centroid_l_l = None
        self.obj_id = '%s_%s' % (self.type, self.specific_type)
        self.max_iter = config['dataset_partition']['max_iter']
        self.model = None
        self.signature = None

    def get_model(self, config):
        return multiple_kmeans.KMeans(config)

    def _preprocess(self, base):
        # 得到m个kmeans的质心
        self.model.fit(base)
        # mk * d
        centroids = self.model.cluster_centers_

        # 用于测试查看初始状态label的分布
        # self.centroids = centroids
        # self.get_label(self.model.labels_)
        # print("centroids", centroids[:, :2])

        centroid_sort_idx = self.random_projection(centroids)
        # random_projection将数组进行排序, 使其符合k组, 每组m个的形式
        # centroid_sort_idx = centroid_sort_idx.reshape(self.n_instance, -1)
        centroid_sort_idx = centroid_sort_idx.reshape(self.n_cluster, -1)

        # 这里处于计算考虑, 不再搞随机抽取
        # total_permutation = self.get_total_permutation()

        # 在这k组中每组各提取一个向量, 一个模型的质心
        centroid_l_l = []
        for i in range(self.n_instance):
            model_centroid_l = []
            for j in range(self.n_cluster):
                idx = centroid_sort_idx[j][i]
                # idx = centroid_sort_idx[j][total_permutation[j][i]]
                model_centroid_l.append(centroids[idx])
            centroid_l_l.append(model_centroid_l)
        self.centroid_l_l = np.array(centroid_l_l)

        # 将这些质心分配给各个模型实例
        for i in range(self.n_instance):
            # print("actual centroids", self.centroid_l_l[i][:, :2])
            self.model_l[i].get_centroid(self.centroid_l_l[i])
        return self.model_l

    # 得到group后生成随机抽取顺序
    def get_total_permutation(self):
        # 随机抽取
        total_permutation = None  # n_cluster * n_instance
        for i in range(self.n_cluster):
            arr = np.random.permutation(self.n_instance)
            arr = np.array([arr])
            if i == 0:
                total_permutation = arr
            else:
                total_permutation = np.append(total_permutation, arr, axis=0)
        return total_permutation

    def random_projection(self, centroid_l):
        res_idx = np.arange(self.n_cluster * self.n_instance)
        BaseMultipleKMeans.divide_and_conquer(0, self.n_cluster, centroid_l, 0, len(centroid_l), res_idx)
        return res_idx

    @staticmethod
    def divide_and_conquer(depth, k, centroid_l, start, end, res_idx):
        if 2 ** depth == k:
            return
        # vector = np.random.rand(centroid_l.shape[1])
        # random_vector = vector / np.linalg.norm(vector)
        random_vector = np.random.normal(size=centroid_l.shape[1], scale=100)
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
        BaseMultipleKMeans.divide_and_conquer(depth, k, centroid_l, start, mid, res_idx)
        BaseMultipleKMeans.divide_and_conquer(depth, k, centroid_l, mid, end, res_idx)

    # 用于测试查看不同点的分布情况
    def get_label(self, labels):
        self.n_point_label = []
        self.label_map = {}
        for cluster_i in range(self.n_cluster * self.n_instance):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i
            self.n_point_label.append(len(base_idx_i))
