import numpy as np
import os
from util import dir_io


class BaseGraph:
    def __init__(self, config):
        self.type = config['type']
        # 保存该模型参数的地址
        self.save_dir = config['save_dir']
        self.classifier_number = config['classifier_number']
        self.graph = None

    def build_graph(self, base):
        pass

    def save(self):
        # self.graph是数组包着数组
        vertices = len(self.graph)
        edges = 0
        for vecs in self.graph:
            edges += len(vecs)
        assert edges % 2 == 0
        edges = edges / 2

        save_dir = '%s/graph.graph' % self.save_dir
        dir_io.save_file(save_dir)
        with open(save_dir, 'w') as f:
            f.write("%d %d\n" % (vertices, edges))
            for nearest_index in self.graph:
                row_index = ""
                for item in nearest_index:
                    row_index += str(item) + " "
                # print(row_index)
                f.write(row_index + '\n')
        print("graph building complete")

    def __str__(self):
        return 'Build graph %s_%d' % (self.type, self.classifier_number)
