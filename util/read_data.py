import numpy as np
import json


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


# dire指的是当前分类器的train_para路径
def get_score_table_intermediate_config(dire):
    long_term_config_dir = '%s/config/long_term_config.json' % dire
    short_term_config_dir = '%s/config/short_term_config.json' % dire
    short_term_config_before_run_dir = '%s/config/short_term_config_before_run.json' % dire
    intermediate_result_dir = '%s/intermediate_result.json' % dire
    total_score_table_dir = '%s/total_score_table.txt' % dire
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_before_run_dir, 'r') as f:
        short_term_config_before_run = json.load(f)
    with open(intermediate_result_dir, 'r') as f:
        intermediate_result = json.load(f)
    total_score_table = np.loadtxt(total_score_table_dir)
    return long_term_config, short_term_config, short_term_config_before_run, intermediate_result, total_score_table
