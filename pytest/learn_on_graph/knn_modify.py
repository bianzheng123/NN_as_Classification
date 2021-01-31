import _init_paths
from procedure_nn_classification.dataset_partition_1.learn_on_graph.build_graph import knn, hnsw
from util.vecs import vecs_io
import numpy as np
from time import time
import os


def build_graph(label, graph, fname):
    start_time = time()
    base_dir = '/home/bianzheng/NN_as_Classification/data/siftsmall_10/base.npy'
    base = np.load(base_dir)
    base = base.astype(np.float32)
    config = {
        "type": "knn_modify",
        "k_graph": 10,
        "save_dir": "/home/bianzheng/NN_as_Classification/pytest/data/graph_data",
        "classifier_number": 1,
        'increase_weight': 2
    }

    graph_ins = knn.KNN(config)
    graph_ins.build_graph(base, label, graph)
    graph = graph_ins.save(fname)
    end_time = time()
    print("time to build graph", (end_time - start_time))
    return graph


def graph_partition(fname, partition_fname):
    start_time = time()
    save_dir = '/home/bianzheng/NN_as_Classification/pytest/data/graph_data'
    command = "/home/bianzheng/software/KaHIP/deploy/kaffpa %s/%s --output_filename=%s/%s --preconfiguration=%s --k=%d" % (
        save_dir, fname, save_dir, partition_fname, 'eco', 16)
    print(command)
    os.system(command)
    # '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt --k=%d'
    end_time = time()
    print("time to partition", (end_time - start_time))


out_label = None
out_graph = None
for i in range(2):
    out_fname = 'graph_%d.graph' % i
    out_partition_fname = 'partition_%d.txt' % i
    out_graph = build_graph(out_label, out_graph, out_fname)
    graph_partition(out_fname, out_partition_fname)
    load_label_dir = '/home/bianzheng/NN_as_Classification/pytest/data/graph_data/' + out_partition_fname
    out_label = np.loadtxt(load_label_dir, dtype=np.int)


class TestGraph:

    def test_vertices(self):
        assert vertices == len(graph)

    def test_edges(self):
        test_edge = 0
        for each_edge in graph:
            test_edge += len(each_edge)
        assert test_edge / 2 == edges

    def test_self_loop(self):
        for i, row in enumerate(graph, 1):
            assert i not in row

# /home/bz/KaHIP/deploy/graphchecker /home/bz/R-Classifier-Learn-Hash/test/graph.graph
# 检测是否符合规范
