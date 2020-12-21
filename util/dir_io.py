import os
import json
import numpy as np
import torch


def delete_dir_if_exist(dire):
    if os.path.isdir(dire):
        # command = 'sudo rm -rf %s' % dire
        command = 'sudo rm -rf %s' % dire
        print(command)
        os.system(command)


def _save_file(dire):
    os.system('sudo touch %s' % dire)
    os.system('sudo chmod 766 %s' % dire)


def write_ptr(dire):
    _save_file(dire)
    f_ptr = open(dire, "w")
    return f_ptr


def save_torch(obj, dire):
    _save_file(dire)
    torch.save(obj, dire)


def save_array_txt(save_dir, arr, fmt):
    _save_file(save_dir)
    np.savetxt(save_dir, arr, fmt=fmt)


def save_numpy(save_dir, data):
    _save_file(save_dir)
    np.save(save_dir, data)


def save_json(save_dir, result_fname, json_file):
    file_dire = '%s/%s' % (save_dir, result_fname)
    _save_file(file_dire)
    with open(file_dire, 'w') as f:
        json.dump(json_file, f)


def kahip(save_dir, command):
    _save_file(save_dir)
    os.system(command)


def save_graph(save_dir, graph, vertices, edges):
    _save_file(save_dir)
    with open(save_dir, 'w') as f:
        f.write("%d %d\n" % (vertices, edges))
        for nearest_index in graph:
            row_index = ""
            for item in nearest_index:
                row_index += str(item) + " "
            # print(row_index)
            f.write(row_index + '\n')


def save_graph_edge_weight(save_dir, graph, vertices, edges):
    _save_file(save_dir)
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
    os.system('sudo mkdir %s' % save_dir)

    long_config = config['long_term_config']
    short_config = config['short_term_config']
    short_config_before_run = config['short_term_config_before_run']
    intermediate_result = config['intermediate_result']

    save_json(save_dir, 'long_term_config.json', long_config)
    save_json(save_dir, 'short_term_config.json', short_config)
    save_json(save_dir, 'short_term_config_before_run.json', short_config_before_run)
    save_json(config['save_dir'], 'intermediate_result.json', intermediate_result)
    print('save program: %s' % config['program_fname'])


def mkdir(dire):
    os.system("sudo mkdir %s" % dire)
