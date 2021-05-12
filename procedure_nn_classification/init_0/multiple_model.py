import numpy as np
import sklearn.cluster as cls
from procedure_nn_classification.dataset_partition_1 import kmeans, learn_on_graph, hash
import time
import copy
from util import dir_io

'''
the son class should rewrite the funciton of get_model() and preprocess()
get_model() is to return the model of classifier
preprocess is to return the array of the model
'''


class MultipleBasePartition:
    def __init__(self, config):
        self.program_train_para_dir = config['program_train_para_dir']
        self.type = config['dataset_partition']['type']
        self.n_instance = config['n_instance']
        self.n_cluster = config['n_cluster']
        self.kahip_dir = config['kahip_dir']
        self.distance_metric = config['distance_metric']
        self.model_l = []
        for i in range(self.n_instance):
            tmp_config = copy.deepcopy(config['dataset_partition'])
            tmp_config['type'] = self.type
            tmp_config['classifier_number'] = i
            tmp_config['n_cluster'] = self.n_cluster
            tmp_config['kahip_dir'] = self.kahip_dir
            tmp_config['distance_metric'] = self.distance_metric
            tmp_config['save_dir'] = '%s/Classifier_%d' % (
                self.program_train_para_dir, tmp_config['classifier_number'])
            dir_io.mkdir(tmp_config['save_dir'])
            tmp_model = self.get_model(tmp_config)
            self.model_l.append(tmp_model)

    # the son class should set the list of model to self.model_l
    def preprocess(self, base):
        print('start preprocessing %s' % self.type)
        intermediate = self._preprocess(base)
        print('finish preprocessing %s' % self.type)
        return self.model_l, intermediate


class IndependentKMeans(MultipleBasePartition):

    def __init__(self, config):
        super(IndependentKMeans, self).__init__(config)
        if self.distance_metric != 'l2':
            raise Exception("not support distance metric")
        # program_train_para_dir, n_cluster, n_instance, entity_number, models

    def get_model(self, config):
        return kmeans.IndependentKMeans(config)

    def _preprocess(self, base):
        return {}


'''
the son class need to get the object of m-kmeans, that is self.model
'''


class MultipleKMeans(MultipleBasePartition):

    def __init__(self, config):
        super(MultipleKMeans, self).__init__(config)
        if self.distance_metric != 'l2':
            raise Exception("not support distance metric")
        # to construct the centroid of m-kmeans, the shape is m * k * d
        self.centroid_l_l = None
        if 'max_iter' in config['dataset_partition']:
            self.max_iter = config['dataset_partition']['max_iter']
            print("max_iter %d" % self.max_iter)
        else:
            self.max_iter = 40
        self.model = cls.KMeans(n_clusters=self.n_cluster * self.n_instance, init='k-means++',
                                max_iter=self.max_iter)

    def get_model(self, config):
        return kmeans.MultipleKMeans(config)

    def _preprocess(self, base):
        # return the centroid of m-kmeans
        kmeans_start_time = time.time()
        self.model.fit(base)
        kmeans_end_time = time.time()
        # mk * d
        centroids = self.model.cluster_centers_

        # to see the initial distribution of label
        # self.centroids = centroids
        # self.get_label(self.model.labels_)
        # print("centroids", centroids[:, :2])
        rp_start_time = time.time()
        centroid_sort_idx = self.random_projection(centroids)
        rp_end_time = time.time()
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
        intermediate = {
            'kmeans_time': kmeans_end_time - kmeans_start_time,
            'random_projection_time': rp_end_time - rp_start_time
        }
        return intermediate

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
        MultipleKMeans.divide_and_conquer(0, self.n_cluster, centroid_l, 0, len(centroid_l), res_idx)
        return res_idx

    @staticmethod
    def divide_and_conquer(depth, k, centroid_l, start, end, res_idx):
        if 2 ** depth == k:
            return
        # vector = np.random.rand(centroid_l.shape[1])
        # random_vector = vector / np.linalg.norm(vector)
        random_vector = np.random.normal(size=centroid_l.shape[1], scale=100)
        random_l = []
        for i in range(start, end):
            random_num = np.dot(random_vector, centroid_l[res_idx[i]])
            random_l.append(random_num)
        # random_l is the result of dot product of centroid and random vector(follow Gauss distribution)
        random_l = np.array(random_l)
        depth += 1
        sort_indices = np.argsort(random_l) + start
        mid = int((start + end - 1) / 2)
        res_idx[start:end] = sort_indices
        MultipleKMeans.divide_and_conquer(depth, k, centroid_l, start, mid, res_idx)
        MultipleKMeans.divide_and_conquer(depth, k, centroid_l, mid, end, res_idx)

    # to see the distribution of different centroids
    def get_label(self, labels):
        self.n_point_label = []
        self.label_map = {}
        for cluster_i in range(self.n_cluster * self.n_instance):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i
            self.n_point_label.append(len(base_idx_i))


class MultipleLearnOnGraphKMeans(MultipleBasePartition):
    def __init__(self, config):
        super(MultipleLearnOnGraphKMeans, self).__init__(config)
        if self.distance_metric != 'l2':
            raise Exception("not support distance metric")
        # to construct the centroid of m-kmeans, the shape is m * k * d
        self.centroid_l_l = None
        self.kmeans_iter = 40
        if 'kmeans_iter' in config['dataset_partition']['build_graph']:
            self.kmeans_iter = config['dataset_partition']['build_graph']['max_iter']
            print("kmeans_iter %d" % self.kmeans_iter)

        self.partition_iter = 1
        if 'partition_iter' in config['dataset_partition']['build_graph']:
            self.partition_iter = config['dataset_partition']['build_graph']['partition_iter']
            print('partition_iter %d' % self.partition_iter)
        self.n_part = 2 ** self.partition_iter

        self.model = cls.KMeans(n_clusters=self.n_part * self.n_instance, init='k-means++',
                                max_iter=self.kmeans_iter)

    def get_model(self, config):
        return learn_on_graph.LearnOnGraphMultipleKMeans(config)

    def _preprocess(self, base):
        # return the centroid of m-kmeans
        kmeans_start_time = time.time()
        self.model.fit(base)
        kmeans_end_time = time.time()
        # mk * d
        centroids = self.model.cluster_centers_

        # to see the initial distribution of label
        # self.centroids = centroids
        # self.get_label(self.model.labels_)
        # print("centroids", centroids[:, :2])
        rp_start_time = time.time()
        centroid_sort_idx = self.random_projection(centroids)
        rp_end_time = time.time()
        # use random_projection() to sort the array, to fit the shape k * m. k groups with m points in each group
        # centroid_sort_idx = centroid_sort_idx.reshape(self.n_instance, -1)
        centroid_sort_idx = centroid_sort_idx.reshape(self.n_part, -1)

        # for the consideration of complexity, do not use the random extract.
        # total_permutation = self.get_total_permutation()

        # here extract a vector for each group and we get the k vectors. Use the k vectors as the centroid of the model
        centroid_l_l = []
        for i in range(self.n_instance):
            model_centroid_l = []
            for j in range(self.n_part):
                idx = centroid_sort_idx[j][i]
                model_centroid_l.append(centroids[idx])
            centroid_l_l.append(model_centroid_l)
        self.centroid_l_l = np.array(centroid_l_l)

        # assign the centroid to each model instance
        for i in range(self.n_instance):
            # print("actual centroids", self.centroid_l_l[i][:, :2])
            self.model_l[i].get_centroid(self.centroid_l_l[i])
        intermediate = {
            'kmeans_time': kmeans_end_time - kmeans_start_time,
            'random_projection_time': rp_end_time - rp_start_time
        }
        return intermediate

    # generate random permutation after got the group
    def get_total_permutation(self):
        # random select
        total_permutation = None  # n_part * n_instance
        for i in range(self.n_part):
            arr = np.random.permutation(self.n_instance)
            arr = np.array([arr])
            if i == 0:
                total_permutation = arr
            else:
                total_permutation = np.append(total_permutation, arr, axis=0)
        return total_permutation

    def random_projection(self, centroid_l):
        res_idx = np.arange(self.n_part * self.n_instance)
        MultipleKMeans.divide_and_conquer(0, self.n_part, centroid_l, 0, len(centroid_l), res_idx)
        return res_idx

    @staticmethod
    def divide_and_conquer(depth, k, centroid_l, start, end, res_idx):
        if 2 ** depth == k:
            return
        # vector = np.random.rand(centroid_l.shape[1])
        # random_vector = vector / np.linalg.norm(vector)
        random_vector = np.random.normal(size=centroid_l.shape[1], scale=100)
        random_l = []
        for i in range(start, end):
            random_num = np.dot(random_vector, centroid_l[res_idx[i]])
            random_l.append(random_num)
        # random_l is the result of dot product of centroid and random vector(follow Gauss distribution)
        random_l = np.array(random_l)
        depth += 1
        sort_indices = np.argsort(random_l) + start
        mid = int((start + end - 1) / 2)
        res_idx[start:end] = sort_indices
        MultipleKMeans.divide_and_conquer(depth, k, centroid_l, start, mid, res_idx)
        MultipleKMeans.divide_and_conquer(depth, k, centroid_l, mid, end, res_idx)

    # to see the distribution of different centroids
    def get_label(self, labels):
        self.n_point_label = []
        self.label_map = {}
        for cluster_i in range(self.n_part * self.n_instance):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i
            self.n_point_label.append(len(base_idx_i))


class MultipleLearnOnGraph(MultipleBasePartition):

    def __init__(self, config):
        super(MultipleLearnOnGraph, self).__init__(config)

    def get_model(self, config):
        return learn_on_graph.LearnOnGraph(config)

    def _preprocess(self, base):
        return {}


class MultipleHash(MultipleBasePartition):

    def __init__(self, config):
        super(MultipleHash, self).__init__(config)
        if self.distance_metric != 'l2':
            raise Exception("not support distance metric")

    def get_model(self, config):
        if self.type == 'random_hash':
            return hash.RandomHash(config)
        elif self.type == 'lsh':
            return hash.LocalitySensitiveHash(config)
        elif self.type == 'random_projection':
            return hash.RandomProjection(config)
        raise Exception('not support hash type')

    def _preprocess(self, base):
        return {}
