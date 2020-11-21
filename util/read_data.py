import numpy as np


def read_knn_graph(graph_dir):
    with open(graph_dir, 'r') as file:
        lines = file.readlines()

    graph = []
    first_line = lines[0].split(' ')
    vertices = int(first_line[0])
    edges = int(first_line[1])
    for idx, line in enumerate(lines, start=0):
        if idx == 0:
            continue
        line_list = line.split(' ')
        # print(len(line_list))
        line_list = [int(x) for x in line_list if x != '\n']
        graph.append(line_list)
    return graph, vertices, edges


def read_partition(partition_dir):
    with open(partition_dir, 'r') as file:
        lines = file.read().splitlines()

    partition = [int(line) for line in lines]
    return partition


def read_label(label_dir):
    with open(label_dir, 'r') as file:
        lines = file.readlines()

    labels = []
    for line in lines:
        class_y = [float(x) for x in line.split(" ") if x != '\n']
        labels.append(class_y)

    return labels

# data_config = read_config('../config/data_config')
# print(data_config)
# task_config = read_config('../config/task_config')
# print(task_config)
# graph, vertics, edges = read_knn_graph('../data/knn.graph')
# print(graph)
# print(vertics)
# print(edges)
# partition = read_partition('../data/partition.txt')
# print(partition)
# label = read_label(data_config['label_dir'])
# print(label)
# print(type(label[0][0]))
