from procedure_nn_classification.dataset_partition_1 import base_partition
from procedure_nn_classification.dataset_partition_1.build_graph import hnsw, knn, partition_knn
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
        self.graph_partition_type = config['graph_partition']['type']
        self.kahip_dir = config['kahip_dir']
        self.preconfiguration = None
        if 'preconfiguration' in config['graph_partition']:
            self.preconfiguration = config['graph_partition']['preconfiguration']
            print("kahip configuration {} {}".format(self.graph_partition_type, self.preconfiguration))
        if self.preconfiguration is None:
            if self.graph_partition_type == 'kaffpa':
                self.preconfiguration = 'eco'
                # stong eco fast fastsocial ecosocial strongsocial
            elif self.graph_partition_type == 'parhip':
                self.preconfiguration = 'fastsocial'
                # ecosocial fastsocial ultrafastsocial ecomesh fastmesh ultrafastmesh
            else:
                raise Exception("not support graph_partition_type")

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
        if self.type == 'knn_random_projection' or self.type == 'knn_lsh' or self.type == 'knn_kmeans':
            graph_partition_config = {
                "kahip_dir": self.kahip_dir,
                "save_dir": self.save_dir,
                "graph_partition_type": self.graph_partition_type,
                'preconfiguration': self.preconfiguration,
                "n_cluster": self.n_cluster,
                'n_item': len(base)
            }
            self.labels = graph_ins.graph_partition(graph_partition_config)
        else:
            self.graph_partition()
        graph_partition_end_time = time.time()
        self.intermediate['graph_partition_time'] = graph_partition_end_time - graph_partition_start_time
        return graph, self.labels

    def graph_partition(self):
        # this function is to invoke kahip and read partition.txt
        partition_dir = '%s/partition.txt' % self.save_dir
        if self.graph_partition_type == 'kaffpa':
            kahip_command = '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt ' \
                            '--k=%d' % (
                                self.kahip_dir, self.save_dir, self.preconfiguration,
                                self.save_dir,
                                self.n_cluster)
            print(kahip_command)
            dir_io.kahip(partition_dir, kahip_command)
        elif self.graph_partition_type == 'parhip':
            kahip_command = 'mpirun -n %d %s/deploy/parhip %s/graph.graph --preconfiguration %s ' \
                            '--save_partition --k %d' % (
                                multiprocessing.cpu_count() // 2, self.kahip_dir, self.save_dir,
                                self.preconfiguration,
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
        elif _type == 'knn_random_projection':
            if config['distance_metric'] != 'l2':
                raise Exception("not support distance metrics")
            return partition_knn.KNNRandomProjection(config)
        elif _type == 'knn_lsh':
            if config['distance_metric'] != 'l2':
                raise Exception("not support distance metrics")
            return partition_knn.KNNLSH(config)
        elif _type == 'knn_kmeans':
            if config['distance_metric'] != 'l2':
                raise Exception("not support distance metrics")
            return partition_knn.KNNKMeans(config)
        elif _type == 'hnsw':
            if config['distance_metric'] != 'l2':
                raise Exception("not support distance metrics")
            return hnsw.HNSW(config)
        raise Exception('do not support the type of buildin a graph')


class LearnOnGraphMultipleKMeans(base_partition.BasePartition):

    def __init__(self, config):
        super(LearnOnGraphMultipleKMeans, self).__init__(config)
        # self.save_dir, self.classifier_number, self.distance_metric, self.n_cluster,
        # self.labels, self.label_map, self.n_point_label
        self.build_graph_config = config['build_graph']
        self.build_graph_config['distance_metric'] = self.distance_metric
        self.graph_partition_type = config['graph_partition']['type']
        self.kahip_dir = config['kahip_dir']
        self.preconfiguration = None
        if 'preconfiguration' in config['graph_partition']:
            self.preconfiguration = config['graph_partition']['preconfiguration']
            print("kahip configuration {} {}".format(self.graph_partition_type, self.preconfiguration))
        if self.preconfiguration is None:
            if self.graph_partition_type == 'kaffpa':
                self.preconfiguration = 'eco'
                # stong eco fast fastsocial ecosocial strongsocial
            elif self.graph_partition_type == 'parhip':
                self.preconfiguration = 'fastsocial'
                # ecosocial fastsocial ultrafastsocial ecomesh fastmesh ultrafastmesh
            else:
                raise Exception("not support graph_partition_type")

    def get_centroid(self, centroid_l):
        # (2**self.partition_iter) * d
        self.centroid_l = centroid_l

    def _partition(self, base, base_base_gnd, ins_intermediate):
        graph_ins = self.graph_factory(self.type, self.build_graph_config)
        graph_ins.get_centroid(self.centroid_l)
        build_graph_start_time = time.time()
        graph = graph_ins.build_graph(base, base_base_gnd, ins_intermediate)
        build_graph_end_time = time.time()
        self.intermediate['bulid_graph_time'] = build_graph_end_time - build_graph_start_time
        save_graph_start_time = time.time()
        graph_ins.save(graph, self.save_dir)
        save_graph_end_time = time.time()
        self.intermediate['save_graph_time'] = save_graph_end_time - save_graph_start_time
        graph_partition_start_time = time.time()
        graph_partition_config = {
            "kahip_dir": self.kahip_dir,
            "save_dir": self.save_dir,
            "graph_partition_type": self.graph_partition_type,
            'preconfiguration': self.preconfiguration,
            "n_cluster": self.n_cluster,
            'n_item': len(base)
        }
        self.labels = graph_ins.graph_partition(graph_partition_config)
        graph_partition_end_time = time.time()
        self.intermediate['graph_partition_time'] = graph_partition_end_time - graph_partition_start_time
        return graph, self.labels

    def graph_partition(self):
        # this function is to invoke kahip and read partition.txt
        partition_dir = '%s/partition.txt' % self.save_dir
        if self.graph_partition_type == 'kaffpa':
            kahip_command = '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt ' \
                            '--k=%d' % (
                                self.kahip_dir, self.save_dir, self.preconfiguration,
                                self.save_dir,
                                self.n_cluster)
            print(kahip_command)
            dir_io.kahip(partition_dir, kahip_command)
        elif self.graph_partition_type == 'parhip':
            kahip_command = 'mpirun -n %d %s/deploy/parhip %s/graph.graph --preconfiguration %s ' \
                            '--save_partition --k %d' % (
                                multiprocessing.cpu_count() // 2, self.kahip_dir, self.save_dir, self.preconfiguration,
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
        if _type == 'knn_kmeans_multiple':
            if config['distance_metric'] != 'l2':
                raise Exception("not support distance metrics")
            return partition_knn.KNNMultipleKMeans(config)
        raise Exception('do not support the type of buildin a graph')
