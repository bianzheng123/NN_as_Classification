import os
import json
import numpy as np
import torch


def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        command = 'rm -rf %s' % dire
        print(command)
        os.system(command)


def mkdir(dire):
    os.system("mkdir %s" % dire)


def move_file(old_dir, new_dir):
    os.system("mv %s %s" % (old_dir, new_dir))


def save_pytorch(pth_ins, save_dir):
    torch.save(pth_ins, save_dir)


def write_ptr(dire):
    f_ptr = open(dire, "w")
    return f_ptr


def save_torch(obj, dire):
    torch.save(obj, dire)


def save_array_txt(save_dir, arr, fmt):
    np.savetxt(save_dir, arr, fmt=fmt)


def save_numpy(save_dir, data):
    np.save(save_dir, data)


def save_json(save_dir, result_fname, json_file):
    file_dire = '%s/%s' % (save_dir, result_fname)
    with open(file_dire, 'w') as f:
        json.dump(json_file, f)


def kahip(save_dir, command):
    os.system(command)


def save_graph(save_dir, graph, vertices, edges):
    with open(save_dir, 'w') as f:
        f.write("%d %d\n" % (vertices, edges))
        for nearest_index in graph:
            row_index = ""
            for item in nearest_index:
                row_index += str(item) + " "
            # print(row_index)
            f.write(row_index + '\n')


def save_graph_edge_weight(save_dir, graph, vertices, edges):
    with open(save_dir, 'w') as f:
        f.write("%d %d 1\n" % (vertices, edges))
        for nearest_index in graph:
            row_index = ""
            for item in nearest_index:
                row_index += str(item) + " " + str(nearest_index[item]) + " "
            # print(row_index)
            f.write(row_index + '\n')


def save_config(config):
    save_dir = '%s/config' % config['save_dir']
    mkdir(save_dir)

    long_config = config['long_term_config']
    short_config = config['short_term_config']
    short_config_before_run = config['short_term_config_before_run']
    intermediate_result = config['intermediate_result']

    save_json(save_dir, 'long_term_config.json', long_config)
    save_json(save_dir, 'short_term_config.json', short_config)
    save_json(save_dir, 'short_term_config_before_run.json', short_config_before_run)
    save_json(config['save_dir'], 'intermediate_result.json', intermediate_result)
    print('save program: %s' % config['program_fname'])
