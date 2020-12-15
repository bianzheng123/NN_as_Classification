from procedure_counting_index.dataset_partition_1 import base_partition
import numpy as np


class KMeansMultiple(base_partition.BasePartition):

    def __init__(self, config):
        super(KMeansMultiple, self).__init__(config)
        self.centroid_l = None
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels

    def get_centroid(self, centroid_l):
        # k * d
        self.centroid_l = centroid_l

    def _partition(self, base):
        # count the distance for each item and centroid to get the distance_table
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

    def _predict(self, query):
        pred = np.array([])
        for item in query:
            # compute the distance between item and the centroid
            distance_table = [np.linalg.norm(item - centroid) for centroid in self.centroid_l]
            nearest_idx = np.argmin(distance_table)
            pred = np.append(pred, np.array([nearest_idx]))
        return pred
