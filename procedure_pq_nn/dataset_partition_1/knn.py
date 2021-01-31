import numpy as np
from procedure_nn_classification.dataset_partition_1.build_graph import knn
import faiss
from util import dir_io
import faiss


class KNN(knn.KNN):
    def __init__(self, config):
        if 'k_graph' in config:
            self.k_graph = config['k_graph']
        else:
            self.k_graph = 50

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base, base_base_gnd, ins_intermediate):
        vertices = len(base)
        if vertices < self.k_graph + 1:
            print("build graph error, input dataset is too samll, do not meet the demand of number of edge")
            return

        dim = base.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(base)
        distance, index_arr = index.search(base, self.k_graph + 1)  # first index should be it self, so +1
        index_arr = index_arr[:, :] + 1  # kahip require the order start from 1, so add 1 in total
        weightless_graph = index_arr.tolist()
        for i in range(len(weightless_graph)):
            weightless_graph[i] = set(weightless_graph[i])

        # print("get the nearest k result")

        for i in range(len(weightless_graph)):
            if (i + 1) in weightless_graph[i]:
                weightless_graph[i].remove((i + 1))
            for vertices_index in weightless_graph[i]:
                if (i + 1) not in weightless_graph[vertices_index - 1]:
                    weightless_graph[vertices_index - 1].add(i + 1)

        res_graph = []
        for i in range(len(weightless_graph)):
            tmp_line = {}
            for vertices in weightless_graph[i]:
                tmp_line[vertices] = 1
            res_graph.append(tmp_line)
        # print("change the rank into graph successfully")
        return res_graph
