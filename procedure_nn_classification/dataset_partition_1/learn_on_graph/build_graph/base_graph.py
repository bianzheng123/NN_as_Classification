import numpy as np
from util import dir_io


class BaseGraph:
    def __init__(self, config):
        self.type = config['type']
        self.save_dir = config['save_dir']
        self.classifier_number = config['classifier_number']
        self.graph = None

    def build_graph(self, base, obj):
        pass

    def save(self):
        # self.graph is the 2d array
        vertices = len(self.graph)
        edges = 0
        for vecs in self.graph:
            edges += len(vecs)
        assert edges % 2 == 0
        edges = edges / 2

        save_dir = '%s/graph.graph' % self.save_dir
        dir_io.save_graph(save_dir, self.graph, vertices, edges)
        print("graph building complete")

    def __str__(self):
        return 'Build graph %s_%d' % (self.type, self.classifier_number)
