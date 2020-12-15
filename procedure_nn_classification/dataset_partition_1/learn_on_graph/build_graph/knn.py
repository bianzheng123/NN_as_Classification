from procedure_nn_classification.dataset_partition_1.learn_on_graph.build_graph import base_graph
import numpy as np
import faiss


class KNN(base_graph.BaseGraph):

    def __init__(self, config):
        super(KNN, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.graph = None
        self.k_graph = config['k_graph']

    '''
    input base
    output vertices and edge
    '''

    def build_graph(self, base):
        vertices = len(base)
        if vertices < self.k_graph + 1:
            raise Exception("build graph error, input dataset is too samll, do not meet the demand of number of edge")

        dim = base.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(base)
        distance, index_arr = index.search(base, self.k_graph + 1)  # +1 because the first index must be itself
        index_arr = index_arr[:, :] + 1  # kahip need the index start from 1, so +1
        result_graph = index_arr.tolist()
        # print("get the nearest k result")
        for i in range(len(result_graph)):
            result_graph[i] = set(result_graph[i])

        for i in range(len(result_graph)):
            if (i + 1) in result_graph[i]:
                result_graph[i].remove((i + 1))
            for vertices_index in result_graph[i]:
                if (i + 1) not in result_graph[vertices_index - 1]:
                    result_graph[vertices_index - 1].add(i + 1)
        # print("change the rank into graph successfully")

        self.graph = result_graph
