from procedure_nn_classification.dataset_partition_1 import base_partition
import numpy as np
import time


class KMeans(base_partition.BasePartition):

    def __init__(self, config):
        super(KMeans, self).__init__(config)
        self.centroid_l = None
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def get_centroid(self, centroid_l):
        # k * d
        self.centroid_l = centroid_l

    def _partition(self, base, obj):
        # count the distance for each item and centroid to get the distance_table
        count_label_start_time = time.time()
        distance_table = None
        for i, vecs in enumerate(base, 0):
            if i == 0:
                distance_table = [np.linalg.norm(base[0] - centroid) for centroid in self.centroid_l]
                distance_table = np.array([distance_table])
                continue
            tmp_dis = [np.linalg.norm(vecs - centroid) for centroid in self.centroid_l]
            tmp_dis = np.array([tmp_dis])
            distance_table = np.append(distance_table, tmp_dis, axis=0)
        # print(distance_table.shape)
        # get the nearest centroid and use it as the label
        self.labels = np.argmin(distance_table, axis=1)
        count_label_end_time = time.time()
        self.intermediate['count_label_time'] = count_label_end_time - count_label_start_time
