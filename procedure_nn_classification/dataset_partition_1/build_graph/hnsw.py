import numpy as np
import faiss
from util import dir_io


class HNSW:

    def __init__(self, config):
        self.k_graph = 10

    '''
    input the base file
    output the point, edge
    '''

    def build_graph(self, base, base_base_gnd, ins_intermediate):
        vertices = len(base)
        if vertices < self.k_graph + 1:
            raise Exception("build graph error, input dataset is too samll, do not meet the demand of number of edge")

        shuffle_base, permutation_reverse = self.shuffle_base(base, vertices)

        index = faiss.index_factory(int(base.shape[1]), "HNSW" + str(self.k_graph))
        index.add(shuffle_base)
        result_graph = HNSW.get_graph(index.hnsw, permutation_reverse)

        return result_graph

    @staticmethod
    def shuffle_base(base, vertices):
        # this variable means i should insert into permutation_insert[i] during shuffle
        permutation_insert = np.random.permutation(vertices)
        # after shuffle, to restore, current i should insert into permutation_reverse[i]
        permutation_reverse = np.zeros(vertices).astype(np.int)
        for i in range(vertices):
            permutation_reverse[permutation_insert[i]] = i

        res_base = np.zeros(base.shape).astype(np.float32)
        for i in range(vertices):
            vecs = base[i]
            for j in range(base.shape[1]):
                res_base[permutation_insert[i]][j] = vecs[j]
        return res_base, permutation_reverse

    @staticmethod
    def get_graph(hnsw, permutation_reverse):
        level = 0
        graph = []
        for i in range(hnsw.levels.size()):
            be = np.empty(2, 'uint64')
            hnsw.neighbor_range(i, level, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))
            tmp_neighbors = [hnsw.neighbors.at(j) for j in range(be[0], be[1])]
            for j in range(len(tmp_neighbors)):
                if tmp_neighbors[j] != 0:
                    tmp_neighbors[j] = permutation_reverse[tmp_neighbors[j]]
                tmp_neighbors[j] += 1
            tmp_neighbors = set(tmp_neighbors)
            if 0 in tmp_neighbors:
                tmp_neighbors.remove(0)
            graph.append(tmp_neighbors)

        for i in range(len(graph)):
            if (i + 1) in graph[i]:
                graph[i].remove((i + 1))
            for vertices_index in graph[i]:
                if (i + 1) not in graph[vertices_index - 1]:
                    graph[vertices_index - 1].add(i + 1)
        return graph

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
        dir_io.save_graph(save_dir, graph, vertices, edges)
        print("graph building complete")
