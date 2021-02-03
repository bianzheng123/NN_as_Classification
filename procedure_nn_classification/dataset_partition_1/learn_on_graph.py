from procedure_nn_classification.dataset_partition_1 import base_partition
from procedure_nn_classification.dataset_partition_1.build_graph import hnsw, knn
import numpy as np
from util import read_data, dir_io
import time
import os
import multiprocessing


class LearnOnGraph(base_partition.BasePartition):

    def __init__(self, config):
        super(LearnOnGraph, self).__init__(config)
        # self.save_dir, self.classifier_number, self.distance_metric, self.n_cluster,
        # self.labels, self.label_map, self.n_point_label
        self.build_graph_config = config['build_graph']
        self.build_graph_config['distance_metric'] = self.distance_metric
        self.graph_partition_type = config['graph_partition']
        self.kahip_dir = config['kahip_dir']

    def _partition(self, base, base_base_gnd, ins_intermediate):
        graph_ins = self.graph_factory(self.type, self.build_graph_config)
        build_graph_start_time = time.time()
        graph = graph_ins.build_graph(base, base_base_gnd, ins_intermediate)
        build_graph_end_time = time.time()
        self.intermediate['bulid_graph_time'] = build_graph_end_time - build_graph_start_time
        save_graph_start_time = time.time()
        graph_ins.save(graph, self.save_dir)
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
        if self.graph_partition_type == 'kaffpa':
            kahip_command = '%s/deploy/kaffpa %s/graph.graph --preconfiguration=eco --output_filename=%s/partition.txt ' \
                            '--k=%d' % (
                                self.kahip_dir, self.save_dir,
                                self.save_dir,
                                self.n_cluster)
            print(kahip_command)
            dir_io.kahip(partition_dir, kahip_command)
        elif self.graph_partition_type == 'parhip':
            kahip_command = 'mpirun -n %d %s/deploy/parhip %s/graph.graph --preconfiguration fastsocial ' \
                            '--save_partition --k %d' % (
                                multiprocessing.cpu_count() // 2, self.kahip_dir, self.save_dir,
                                self.n_cluster)
            print(kahip_command)
            dir_io.kahip('./tmppartition.txtp', kahip_command)
            self.move_partition_txt()
        labels = read_data.read_partition(partition_dir)
        self.labels = np.array(labels)

    def move_partition_txt(self):
        dir_io.move_file('tmppartition.txtp', '%s/partition.txt' % self.save_dir)

    @staticmethod
    def graph_factory(_type, config):
        if _type == 'knn':
            return knn.KNN(config)
        elif _type == 'hnsw':
            if config['distance_metric'] != 'l2':
                raise Exception("not support distance metrics")
            return hnsw.HNSW(config)
        raise Exception('do not support the type of buildin a graph')