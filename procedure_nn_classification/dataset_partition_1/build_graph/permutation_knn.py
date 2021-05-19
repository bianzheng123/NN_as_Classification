import numpy as np
import faiss
from util import dir_io


class KNN:
    def __init__(self, config):
        pass

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base, base_base_gnd, k_idx_l):

        vertices = len(base)

        index_arr = base_base_gnd[:, k_idx_l]  # +1 because the first index must be itself
        # print(index_arr.shape)
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
        return res_graph

    @staticmethod
    def save(graph, save_dir):
        # graph is the 2d array
        vertices = len(graph)
        edges = 0
        for vecs in graph:
            edges += len(vecs)
        assert edges % 2 == 0
        edges = edges / 2

        save_dir = '%s/graph.graph' % save_dir
        dir_io.save_graph_edge_weight(save_dir, graph, vertices, edges)
        print("save graph complete")
