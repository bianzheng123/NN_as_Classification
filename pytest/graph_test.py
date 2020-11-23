import _init_paths
from procedure.dataset_partition_1.learn_on_graph.build_graph import knn, hnsw
from util.vecs import vecs_io
import numpy as np
from time import time
import os
import faiss


def build_graph():
    start_time = time()
    base_dir = '/home/bz/SIFT/siftsmall/siftsmall_base.fvecs'
    base = vecs_io.fvecs_read_mmap(base_dir)[0]
    base = base.astype(np.float32)
    config = {
        "type": "knn",
        "k_graph": 10,
        "save_dir": "/home/bz/NN_as_Classification/pytest",
        "classifier_number": 1
    }

    graph_ins = hnsw.HNSW(config)
    graph_ins.build_graph(base)
    graph_ins.save()
    end_time = time()
    print("time to build graph", (end_time - start_time))


def graph_partition():
    start_time = time()
    save_dir = '/home/bz/NN_as_Classification/pytest'
    os.system(
        "/home/bz/KaHIP/deploy/kaffpa %s/graph.graph --output_filename=%s/partition.txt --preconfiguration=%s --k=%d" % (
            save_dir, save_dir, 'eco', 16))
    # '%s/deploy/kaffpa %s/graph.graph --preconfiguration=%s --output_filename=%s/partition.txt --k=%d'
    end_time = time()
    print("time to partition", (end_time - start_time))


build_graph()
graph_partition()


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
