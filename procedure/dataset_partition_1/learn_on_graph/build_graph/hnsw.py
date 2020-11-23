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

        index = faiss.index_factory(int(base.shape[1]), "HNSW" + str(self.k_graph))
        index.add(base)
        result_graph = HNSW.get_graph(index.hnsw)

        self.graph = result_graph

    @staticmethod
    def get_graph(hnsw):
        level = 0
        graph = []
        for i in range(hnsw.levels.size()):
            be = np.empty(2, 'uint64')
            hnsw.neighbor_range(i, level, faiss.swig_ptr(be), faiss.swig_ptr(be[1:]))
            tmp_neighbors = [hnsw.neighbors.at(j) for j in range(be[0], be[1])]
            # print(type(tmp_neighbors))
            for j in range(len(tmp_neighbors)):
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
