from procedure.dataset_partition_1 import base_partition
from procedure.dataset_partition_1.learn_on_graph.build_graph import knn, hnsw
import numpy as np
import os
from util import read_data, dir_io


class LearnOnGraph(base_partition.BasePartition):

    def __init__(self, config):
        super(LearnOnGraph, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels
        config['build_graph']['save_dir'] = self.save_dir
        config['build_graph']['classifier_number'] = self.classifier_number
        self.build_graph_config = config['build_graph']
        self.graph_partition_config = config['graph_partition']
        self.kahip_dir = config['kahip_dir']

    def _partition(self, base):
        graph_ins = graph_factory(self.build_graph_config)
        graph_ins.build_graph(base)
        graph_ins.save()
        self.graph_partition()

    def graph_partition(self):
        # this function is to invoke kahip and read partition.txt
        dir_io.save_file('%s/partition.txt' % self.save_dir)
        kahip_command = '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt ' \
                        '--k=%d --time_limit=%d' % (
                            self.kahip_dir, self.save_dir, self.graph_partition_config['preconfiguration'],
                            self.save_dir,
                            self.n_cluster, self.graph_partition_config['time_limit'])
        os.system(kahip_command)
        partition_dir = '%s/partition.txt' % self.save_dir
        labels = read_data.read_partition(partition_dir)
        self.labels = np.array(labels)


def graph_factory(config):
    _type = config['type']
    if _type == 'knn':
        return knn.KNN(config)
    elif _type == 'hnsw':
        return hnsw.HNSW(config)
    raise Exception('do not support the type of buildin a graph')
