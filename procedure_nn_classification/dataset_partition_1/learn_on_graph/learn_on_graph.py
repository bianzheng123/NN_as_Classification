from procedure_nn_classification.dataset_partition_1 import base_partition
from procedure_nn_classification.dataset_partition_1.learn_on_graph.build_graph import knn, hnsw
import numpy as np
from util import read_data, dir_io
import time
import os


class LearnOnGraph(base_partition.BasePartition):

    def __init__(self, config):
        super(LearnOnGraph, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels
        config['build_graph']['save_dir'] = self.save_dir
        config['build_graph']['classifier_number'] = self.classifier_number
        self.build_graph_config = config['build_graph']
        self.graph_partition_config = config['graph_partition']
        self.kahip_dir = config['kahip_dir']
        self.n_process = 8

    def _partition(self, base, obj):
        graph_ins = graph_factory(self.build_graph_config)
        build_graph_start_time = time.time()
        graph_ins.build_graph(base, obj)
        build_graph_end_time = time.time()
        self.intermediate['bulid_graph_time'] = build_graph_end_time - build_graph_start_time
        save_graph_start_time = time.time()
        graph = graph_ins.save()
        save_graph_end_time = time.time()
        self.intermediate['save_graph_time'] = save_graph_end_time - save_graph_start_time
        graph_partition_start_time = time.time()
        self.graph_partition()
        graph_partition_end_time = time.time()
        self.intermediate['graph_partition_time'] = graph_partition_end_time - graph_partition_start_time
        return graph, self.labels

    def graph_partition(self):
        # this function is to invoke kahip and read partition.txt
        partition_dir = '%s/partition.txt' % self.save_dir
        if self.graph_partition_config['type'] == 'kaffpa':
            kahip_command = '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt ' \
                            '--k=%d' % (
                                self.kahip_dir, self.save_dir, self.graph_partition_config['preconfiguration'],
                                self.save_dir,
                                self.n_cluster)
            print(kahip_command)
            dir_io.kahip(partition_dir, kahip_command)
        elif self.graph_partition_config['type'] == 'parhip':
            kahip_command = 'mpirun -n %d %s/deploy/parhip %s/graph.graph --preconfiguration %s --save_partition --k %d' % (
                self.n_process, self.kahip_dir, self.save_dir, self.graph_partition_config['preconfiguration'],
                self.n_cluster)
            print(kahip_command)
            dir_io.kahip('./tmppartition.txtp', kahip_command)
            self.move_partition_txt()
        labels = read_data.read_partition(partition_dir)
        self.labels = np.array(labels)

    def move_partition_txt(self):
        dir_io.move_file('tmppartition.txtp', '%s/partition.txt' % self.save_dir)


def graph_factory(config):
    _type = config['type']
    if _type == 'knn':
        return knn.KNN(config)
    elif _type == 'hnsw':
        return hnsw.HNSW(config)
    raise Exception('do not support the type of buildin a graph')
