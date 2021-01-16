import numpy as np
import time
from util import dir_io


class BasePartition:
    def __init__(self, config):
        self.type = config['type']
        self.save_dir = '%s/dataset_partition' % config['save_dir']
        dir_io.mkdir(self.save_dir)
        self.classifier_number = config['classifier_number']
        self.entity_number = config['entity_number']
        self.distance_metric = config['distance_metric']
        self.obj_id = "%s_%d_%d" % (self.type, self.entity_number, self.classifier_number)
        # number of cluster
        self.n_cluster = config['n_cluster']
        self.model_info = None
        # the key of map is the number of every class, its value is the index that belongs to the cluster in base
        self.label_map = []
        # to count the number of points in different bucket
        self.n_point_label = None
        self.intermediate = {}

        self.labels = None

    def partition(self, base, base_base_gnd, ins_intermediate):
        start_time = time.time()
        print('start dataset partitioning %s' % self.obj_id)
        para = self._partition(base, base_base_gnd, ins_intermediate)
        self.get_labels(self.labels)
        print('finish dataset partitioning %s' % self.obj_id)
        end_time = time.time()
        self.intermediate['total_time'] = end_time - start_time
        self.intermediate['cluster_number_distribution'] = self.n_point_label
        model_info = {
            "classifier_number": self.classifier_number,
            "entity_number": self.entity_number,
        }
        return (self.labels, self.label_map), (model_info, self.intermediate), para

    '''
    son class should get the self.labels, which is a list, the label for each item
    '''

    def _partition(self, base, base_base_gnd, ins_intermediate):
        pass

    # the function partition the base according to the number of cluster
    # the labels should be the array of numpy
    def get_labels(self, labels):
        self.n_point_label = []
        for cluster_i in range(self.n_cluster):
            base_idx_i = np.argwhere(labels == cluster_i).reshape(-1)
            self.label_map.append(base_idx_i)
            self.n_point_label.append(len(base_idx_i))

    def save(self):
        save_label_dir = '%s/partition.txt' % self.save_dir
        dir_io.save_array_txt(save_label_dir, self.labels, '%i')

        # save_distribution_dir = '%s/distribution_partition.txt' % self.save_dir
        # dir_io.save_array_txt(save_distribution_dir, self.n_point_label, '%i')

    def __str__(self):
        return '%s, n_cluster: %d' % (self.obj_id, self.n_cluster)
