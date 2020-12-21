from procedure_nn_classification.dataset_partition_1.learn_on_graph.build_graph import base_graph
import numpy as np
import faiss
from util import dir_io


class KNN(base_graph.BaseGraph):

    def __init__(self, config):
        super(KNN, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.graph = None
        self.k_graph = config['k_graph']
        self.increase_weight = config['increase_weight']

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base, obj):
        if obj is not None:
            graph, label = obj
            for i, cls in enumerate(label, 0):
                for neighbor in graph[i]:
                    if label[neighbor - 1] != cls:  # means the neighbor have different cluster
                        graph[i][neighbor] = graph[i][neighbor] + self.increase_weight
                        graph[neighbor - 1][i + 1] = graph[neighbor - 1][i + 1] + self.increase_weight

            self.graph = graph
            return

        vertices = len(base)
        if vertices < self.k_graph + 1:
            raise Exception("build graph error, input dataset is too samll, do not meet the demand of number of edge")

        dim = base.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(base)
        distance, index_arr = index.search(base, self.k_graph + 1)  # +1 because the first index must be itself
        index_arr = index_arr[:, :] + 1  # kahip need the index start from 1, so +1
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

        self.graph = res_graph

    def save(self):
        # self.graph is the 2d array
        vertices = len(self.graph)
        edges = 0
        for vecs in self.graph:
            edges += len(vecs)
        assert edges % 2 == 0
        edges = edges / 2

        save_dir = '%s/graph.graph' % self.save_dir
        dir_io.save_graph_edge_weight(save_dir, self.graph, vertices, edges)
        print("graph building complete")
        return self.graph