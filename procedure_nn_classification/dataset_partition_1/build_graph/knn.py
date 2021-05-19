import numpy as np
import faiss
from util import dir_io


class KNN:
    def __init__(self, config):
        if 'k_graph' in config:
            self.k_graph = config['k_graph']
            print("k_graph %d" % self.k_graph)
        else:
            self.k_graph = 40
        if 'increase_weight' in config:
            self.increase_weight = config['increase_weight']
            print("increase_weight %d" % self.increase_weight)
        else:
            self.increase_weight = 500

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base, base_base_gnd, ins_intermediate):
        if ins_intermediate is not None:
            graph, label = ins_intermediate
            for i, cls in enumerate(label, 0):
                for neighbor in graph[i]:
                    if label[neighbor - 1] != cls:  # means the neighbor have different cluster
                        # graph[i][neighbor] = 0
                        # graph[neighbor - 1][i + 1] = 0
                        graph[i][neighbor] = graph[i][neighbor] + self.increase_weight
                        graph[neighbor - 1][i + 1] = graph[neighbor - 1][i + 1] + self.increase_weight

            return graph

        vertices = len(base)
        if vertices < self.k_graph + 1:
            raise Exception("build graph error, input dataset is too samll, do not meet the demand of number of edge")
        if self.k_graph + 1 > base_base_gnd.shape[1]:
            # raise Exception("k_graph + 1 > the length in base_base_gnd, system crash. "
            #                 "please use increase the k in base_base_gnd or decrease k_graph")
            print("\033[32;1m Warning! the length of k_graph + 1 is larger than base_base_gnd could provide \033[0m")
            print("length of base_base_gnd %d" % (base_base_gnd.shape[1]))

        index_arr = base_base_gnd[:, :self.k_graph + 1]  # +1 because the first index must be itself
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
