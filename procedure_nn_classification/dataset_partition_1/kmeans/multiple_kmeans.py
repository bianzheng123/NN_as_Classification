from procedure_nn_classification.dataset_partition_1 import base_partition
import numpy as np
import time
from multiprocessing import Pool, cpu_count


class KMeans(base_partition.BasePartition):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        self.centroid_l = None
        self.n_process = cpu_count()
        self.n_pool_process = cpu_count()
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels, self.distance_metric

    def get_centroid(self, centroid_l):
        # k * d
        self.centroid_l = centroid_l

    def _partition(self, base, base_base_gnd, ins_intermediate):
        # count the distance for each item and centroid to get the distance_table
        labels, time_consumed = self.parallel(base)
        self.labels = labels
        self.intermediate['count_label_time'] = time_consumed

    def parallel(self, data):
        start_time = time.time()
        p = Pool(self.n_pool_process)
        res_l = []
        for i in range(self.n_process):
            res = p.apply_async(KMeans.count_centroid, args=(data, self.centroid_l, i, self.n_process))
            res_l.append(res)

        p.close()
        p.join()
        res_labels = np.zeros(data.shape[0]).astype(np.int)
        for i, res in enumerate(res_l, 0):
            tmp_labels = res.get()
            for j in range(len(tmp_labels)):
                res_labels[i + j * self.n_process] = tmp_labels[j]
        # np.savetxt('label_parallel.txt', res_labels, fmt='%d')
        end_time = time.time()
        time_consumed = end_time - start_time
        return res_labels, time_consumed

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
        return np.array(labels)
