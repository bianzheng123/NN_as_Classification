from procedure.dataset_partition_1.learn_on_graph.build_graph import base_graph
import numpy as np
import faiss


class HNSW(base_graph.BaseGraph):

    def __init__(self, config):
        super(HNSW, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.graph = None
        self.k_graph = config['k_graph']

    '''
    输入base文件
    输出点，边的数量以及加权图
    读取bvecs文件, 使用暴力算出每一个点的最邻近距离, 转换成图, 输出为文本
    '''

    def build_graph(self, base):
        vertices = len(base)
        if vertices < self.k_graph + 1:
            raise Exception("建图错误, 输入数据量太少, 不能满足边的数量")

        shuffle_base, permutation_reverse = self.shuffle_base(base, vertices)

        index = faiss.index_factory(int(base.shape[1]), "HNSW" + str(self.k_graph))
        index.add(shuffle_base)
        result_graph = HNSW.get_graph(index.hnsw, permutation_reverse)

        self.graph = result_graph

    @staticmethod
    def shuffle_base(base, vertices):
        # shuffle时应该i应该插入到permutation_insert[i]的位置
        permutation_insert = np.random.permutation(vertices)
        # shuffle后还原时现在的位置i应该插入到permutation_reverse[i]的位置
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
