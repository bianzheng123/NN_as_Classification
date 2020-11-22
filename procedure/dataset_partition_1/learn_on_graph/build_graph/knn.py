from procedure.dataset_partition_1.learn_on_graph.build_graph import base_graph
import numpy as np
import faiss


class KNN(base_graph.BaseGraph):

    def __init__(self, config):
        super(KNN, self).__init__(config)
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

        dim = base.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(base)
        distance, index_arr = index.search(base, self.k_graph + 1)  # 第一个索引一定是自己, 所以+1
        index_arr = index_arr[:, :] + 1  # kahip要求序号从1开始, 就集体加1
        result_graph = index_arr.tolist()
        print("得到最近的k个结果")
        for i in range(len(result_graph)):
            result_graph[i] = set(result_graph[i])

        for i in range(len(result_graph)):
            if (i + 1) in result_graph[i]:
                result_graph[i].remove((i + 1))
            for vertices_index in result_graph[i]:
                if (i + 1) not in result_graph[vertices_index - 1]:
                    result_graph[vertices_index - 1].add(i + 1)
        print("将rank转换成图")

        self.graph = result_graph
