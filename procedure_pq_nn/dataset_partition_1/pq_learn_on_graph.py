from procedure_nn_classification.dataset_partition_1 import learn_on_graph
from procedure_nn_classification.dataset_partition_1.build_graph import hnsw
from procedure_pq_nn.dataset_partition_1 import knn
import numpy as np
from util import read_data, dir_io
import time
import os


class PQLearnOnGraph(learn_on_graph.LearnOnGraph):

    @staticmethod
    def graph_factory(_type, config):
        if _type == 'knn':
            return knn.KNN(config)
        elif _type == 'hnsw':
            return hnsw.HNSW(config)
        raise Exception('do not support the type of buildin a graph')
