from procedure.dataset_partition_1 import base_partition
from procedure.dataset_partition_1.learn_on_graph.build_graph import knn
import numpy as np
import os
from util import read_data


class LearnOnGraph(base_partition.BasePartition):

    def __init__(self, config):
        super(LearnOnGraph, self).__init__(config)
        # 该模型需要聚类的数量
        # self.type, self.save_dir, self.classifier_number, self.label_map, self.n_cluster, self.labels
        config['build_graph']['save_dir'] = self.save_dir
        config['build_graph']['classifier_number'] = self.classifier_number
        self.build_graph_config = config['build_graph']
        self.graph_partition_config = config['graph_partition']
        self.kahip_dir = config['kahip_dir']

    def partition(self, base):
        print('start training %s_%d' % (self.type, self.classifier_number))
        graph_ins = graph_factory(self.build_graph_config)
        graph_ins.build_graph(base)
        graph_ins.save()
        self.graph_partition()
        self.get_labels(self.labels)
        print('finish training %s_%d' % (self.type, self.classifier_number))
        return self.labels, self.label_map

    def graph_partition(self):
        # 调用kahip, 然后读取partition.txt
        kahip_command = '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt ' \
                        '--k=%d' % (
                            self.kahip_dir, self.save_dir, self.graph_partition_config['preconfiguration'],
                            self.save_dir,
                            self.n_cluster)
        os.system(kahip_command)
        partition_dir = '%s/partition.txt' % self.save_dir
        labels = read_data.read_partition(partition_dir)
        self.labels = np.array(labels)


def graph_factory(config):
    _type = config['type']
    if _type == 'knn':
        return knn.KNN(config)
    raise Exception('建图类型不支持')
