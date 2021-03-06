import numpy as np
from util import dir_io
import copy
import sklearn.cluster as cls
from procedure_counting_index.dataset_partition_1 import partition_model
import time

'''
the son class should rewrite the funciton of get_model() and preprocess()
get_model() is to return the model of classifier
preprocess is to return the array of the model
'''


class PreprocessBasePartition:
    def __init__(self, config):
        self.program_train_para_dir = config['program_train_para_dir']
        self.type = config['type']
        self.n_instance = config['n_instance']
        self.n_cluster = config['n_cluster']
        self.model_l = []
        self.intermediate = {}
        # for identification
        for i in range(self.n_instance):
            tmp_config = copy.deepcopy(config)
            tmp_config['type'] = self.type
            tmp_config['classifier_number'] = i + 1
            tmp_config['n_cluster'] = self.n_cluster
            tmp_config['save_dir'] = '%s/Classifier_%d' % (
                self.program_train_para_dir, tmp_config['classifier_number'])
            dir_io.mkdir(tmp_config['save_dir'])
            tmp_model = self.get_model(tmp_config)
            self.model_l.append(tmp_model)

    # the son class should set the list of model to self.model_l
    def preprocess(self, base):
        print('start preprocessing %s' % self.type)
        self._preprocess(base)
        print('finish preprocessing %s' % self.type)
        return self.model_l, self.intermediate,


class PreprocessKMeansIndependent(PreprocessBasePartition):

    def __init__(self, config):
        super(PreprocessKMeansIndependent, self).__init__(config)
        # program_train_para_dir, n_cluster, n_instance, entity_number, models

    def get_model(self, config):
        return partition_model.KMeans(config)

    def _preprocess(self, base):
        pass


'''
the son class need to get the object of m-kmeans, that is self.model
'''


class PreprocessKMeansMultiple(PreprocessBasePartition):

    def __init__(self, config):
        super(PreprocessKMeansMultiple, self).__init__(config)
        # to construct the centroid of m-kmeans, the shape is m * k * d
        self.centroid_l_l = None
        self.max_iter = config['max_iter']
        self.model = cls.KMeans(n_clusters=self.n_cluster * self.n_instance, init='k-means++',
                                max_iter=self.max_iter)

    def get_model(self, config):
        return partition_model.KMeansMultiple(config)

    def _preprocess(self, base):
        # return the centroid of m-kmeans
        kmeans_start_time = time.time()
        self.model.fit(base)
        kmeans_end_time = time.time()
        self.intermediate['kmeans_time'] = kmeans_end_time - kmeans_start_time

        # mk * d
        centroids = self.model.cluster_centers_

        # to see the initial distribution of label
        # self.centroids = centroids
        # self.get_label(self.model.labels_)
        # print("centroids", centroids[:, :2])
        rp_start_time = time.time()
        centroid_sort_idx = self.random_projection(centroids)
        rp_end_time = time.time()
        self.intermediate['random_projection'] = rp_end_time - rp_start_time
        # use random_projection() to sort the array, to fit the shape k * m. k groups with m points in each group
        # centroid_sort_idx = centroid_sort_idx.reshape(self.n_instance, -1)
        centroid_sort_idx = centroid_sort_idx.reshape(self.n_cluster, -1)

        # for the consideration of complexity, do not use the random extract.
        # total_permutation = self.get_total_permutation()

        # here extract a vector for each group and we get the k vectors. Use the k vectors as the centroid of the model
        centroid_l_l = []
        for i in range(self.n_instance):
            model_centroid_l = []
            for j in range(self.n_cluster):
                idx = centroid_sort_idx[j][i]
                # idx = centroid_sort_idx[j][total_permutation[j][i]]
                model_centroid_l.append(centroids[idx])
            centroid_l_l.append(model_centroid_l)
        self.centroid_l_l = np.array(centroid_l_l)

        # assign the centroid to each model instance
        for i in range(self.n_instance):
            # print("actual centroids", self.centroid_l_l[i][:, :2])
            self.model_l[i].get_centroid(self.centroid_l_l[i])

    # generate random permutation after got the group
    def get_total_permutation(self):
        # random select
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
        PreprocessKMeansMultiple.divide_and_conquer(0, self.n_cluster, centroid_l, 0, len(centroid_l), res_idx)
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
        # random_l is the result of dot product of centroid and random vector(follow Gauss distribution)
        depth += 1
        sort_indices = np.argsort(random_l[start:end]) + start
        mid = int((start + end - 1) / 2)
        res_idx[start:end] = sort_indices
        PreprocessKMeansMultiple.divide_and_conquer(depth, k, centroid_l, start, mid, res_idx)
        PreprocessKMeansMultiple.divide_and_conquer(depth, k, centroid_l, mid, end, res_idx)

    # to see the distribution of different centroids
    def get_label(self, labels):
        self.n_point_label = []
        self.label_map = {}
        for cluster_i in range(self.n_cluster * self.n_instance):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i
            self.n_point_label.append(len(base_idx_i))


class PreprocessLSH(PreprocessBasePartition):

    def __init__(self, config):
        super(PreprocessLSH, self).__init__(config)
        # program_train_para_dir, n_cluster, n_instance, entity_number, models

    def get_model(self, config):
        return partition_model.LSH(config)

    def _preprocess(self, base):
        pass
