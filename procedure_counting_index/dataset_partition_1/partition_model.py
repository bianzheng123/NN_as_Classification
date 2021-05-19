import numpy as np
import time
from util import dir_io
import sklearn.cluster as cls
from multiprocessing import cpu_count, Pool


class BasePartition:
    def __init__(self, config):
        self.type = config['type']
        self.save_dir = '%s/dataset_partition' % config['save_dir']
        dir_io.mkdir(self.save_dir)
        self.classifier_number = config['classifier_number']
        self.obj_id = "%s_%d" % (self.type, self.classifier_number)
        # number of cluster
        self.n_cluster = config['n_cluster']
        self.model_info = None
        # the key of map is the number of every class, its value is the index that belongs to the cluster in base
        self.label_map = {}
        # to count the number of points in different bucket
        self.n_point_label = None

        self.labels = None

    def partition(self, base):
        start_time = time.time()
        print('start dataset partitioning %s' % self.obj_id)
        self._partition(base)
        self.get_labels(self.labels)
        print('finish dataset partitioning %s' % self.obj_id)
        end_time = time.time()
        intermediate_config = {
            'time': end_time - start_time,
            'cluster_number_distribution': self.n_point_label
        }
        return (self.labels, self.label_map), (self.classifier_number, intermediate_config)

    '''
    son class should get the self.labels, which is a list, the label for each item
    '''

    def _partition(self, base):
        pass

    def predict(self, query):
        start_time = time.time()
        print('start predict %s' % self.obj_id)
        pred_cluster = self._predict(query)
        print('finish predict %s' % self.obj_id)
        end_time = time.time()
        intermediate = {
            'time': end_time - start_time,
        }
        return pred_cluster, intermediate

    '''
    in the son class, it should predict the cluster that query belongs to
    query is a 2d array with multiple single query
    '''

    def _predict(self, query):
        pass

    # the function partition the base according to the number of cluster
    # the labels should be the array of numpy
    def get_labels(self, labels):
        self.n_point_label = []
        for cluster_i in range(self.n_cluster):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map[cluster_i] = base_idx_i
            self.n_point_label.append(len(base_idx_i))

    def save(self):
        save_label_dir = '%s/partition.txt' % self.save_dir
        dir_io.save_array_txt(save_label_dir, self.labels, '%i')
        # save_distribution_dir = '%s/distribution_partition.txt' % self.save_dir
        # dir_io.save_array_txt(save_distribution_dir, self.n_point_label, '%i')


class KMeans(BasePartition):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        self.max_iter = config['max_iter']
        self.model = cls.KMeans(n_clusters=self.n_cluster, init='k-means++', max_iter=self.max_iter)
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base):
        self.model.fit(base)
        self.labels = self.model.labels_

    def _predict(self, query):
        pred = self.model.predict(query)
        return pred


class KMeansMultiple(BasePartition):

    def __init__(self, config):
        super(KMeansMultiple, self).__init__(config)
        self.centroid_l = None
        self.n_process = cpu_count() // 10 * 9
        self.n_pool_process = cpu_count() // 10 * 9
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def get_centroid(self, centroid_l):
        # k * d
        self.centroid_l = centroid_l

    def _partition(self, base):
        # count the distance for each item and centroid to get the distance_table
        labels = self.parallel(base)
        self.labels = labels

    def parallel(self, data):
        start_time = time.time()
        p = Pool(self.n_pool_process)
        res_l = []
        for i in range(self.n_process):
            res = p.apply_async(KMeansMultiple.count_centroid, args=(data, self.centroid_l, i, self.n_process))
            res_l.append(res)

        p.close()
        p.join()
        res_labels = np.zeros(data.shape[0]).astype(np.int64)
        for i, res in enumerate(res_l, 0):
            tmp_labels = res.get()
            for j in range(len(tmp_labels)):
                res_labels[i + j * self.n_process] = tmp_labels[j]
        # np.savetxt('label_parallel.txt', res_labels, fmt='%d')
        end_time = time.time()
        return res_labels

    @staticmethod
    def count_centroid(base, centroid_l, idx, pool_size):
        # count the distance for each item and centroid to get the distance_table
        labels = []
        len_base = len(base)
        for i in range(idx, len_base, pool_size):
            vecs = base[i]
            tmp_dis = [np.linalg.norm(vecs - centroid) for centroid in centroid_l]
            tmp_label = np.argmin(tmp_dis, axis=0)
            labels.append(tmp_label)
        return np.array(labels, dtype=np.int64)

    def _predict(self, query):
        pred = np.array([])
        for item in query:
            # compute the distance between item and the centroid
            distance_table = [np.linalg.norm(item - centroid) for centroid in self.centroid_l]
            nearest_idx = np.argmin(distance_table)
            pred = np.append(pred, np.array([nearest_idx]))
        return pred


class LSH(BasePartition):

    def __init__(self, config):
        super(LSH, self).__init__(config)
        self.r = 1
        self.a_sigma = 1
        self.a_miu = 0
        if 'a_sigma' in config:
            self.a_sigma = config['a_sigma']
            print('a_sigma %d' % self.a_sigma)
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def _partition(self, base):
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

    def _predict(self, query):
        query_norm = query / self.norm_div
        proj = np.dot(query_norm, self.a)
        predict = np.floor((proj + self.b) / self.r) % self.n_cluster
        predict = predict.astype(np.int)
        return predict
