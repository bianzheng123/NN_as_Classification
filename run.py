from util.numpy import load_data
from procedure.check_config_0 import check_config
from procedure.dataset_partition_1 import dataset_partition
from procedure.prepare_train_sample_2 import prepare_train_sample
from procedure.train_model_3 import train_model
from procedure.result_integrate_4 import result_integrate
import json
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader, TensorDataset


def delete_dir_if_exist(dir):
    if os.path.isdir(dir):
        command = 'rm -rf %s' % dir
        print(command)
        os.system(command)


def save_result_config(config):
    save_dir = '%s/config' % config['program_result_dir']
    os.system('mkdir %s' % save_dir)
    long_config = config['long_term_config']
    short_config = config['short_term_config']
    short_config_before_run = config['short_term_config_before_run']
    result_integrate.save_json(save_dir, 'long_term_config.json', long_config)
    result_integrate.save_json(save_dir, 'short_term_config.json', short_config)
    result_integrate.save_json(save_dir, 'short_term_config_before_run.json', short_config_before_run)


if __name__ == '__main__':
    long_term_config_dir = 'config/run/long_term_config.json'
    short_term_config_dir = 'config/run/short_term_config.json'

    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    train_para_dir = '%s/train_para/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    delete_dir_if_exist(train_para_dir)
    result_dir = '%s/result/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    delete_dir_if_exist(result_dir)

    data_dir = '%s/data/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    base, query, learn, gnd = load_data.load_data_npy(load_data_config)

    check_config.check_config(short_term_config)

    program_train_para_dir = '%s/train_para/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    dataset_partition_config = {
        'kahip_dir': long_term_config['kahip_dir'],
        "n_classifier": short_term_config['n_classifier'],
        "program_train_para_dir": program_train_para_dir,
        "dataset_partition": short_term_config['dataset_partition'],
        "mutual_attribute": short_term_config['mutual_attribute']
    }
    partition_info_l = dataset_partition.partition(base, dataset_partition_config)

    # print(len(partition_info))
    # print(partition_info[0][0])
    # print(partition_info[0][1])

    prepare_train_config = {
        "n_classifier": short_term_config['n_classifier'],
        'program_train_para_dir': program_train_para_dir,
        'prepare_train': short_term_config['prepare_train_sample'],
        "mutual_attribute": short_term_config['mutual_attribute']
    }
    trainset_l = prepare_train_sample.prepare_train(base, partition_info_l, prepare_train_config)

    # trainloader, valloader = trainset_l[0]
    # print(trainloader.batch_size)
    # print(valloader.batch_size)
    # print(len(trainloader))
    # print(len(valloader))

    train_model_config = {
        "n_classifier": short_term_config['n_classifier'],
        'program_train_para_dir': program_train_para_dir,
        'train_model': short_term_config['train_model'],
        'mutual_attribute': short_term_config['mutual_attribute']
    }
    train_model_ins_l = train_model.train_model(base, trainset_l, train_model_config)

    program_result_dir = '%s/result/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    result_integrate_config = {
        'k': long_term_config['k'],
        'n_classifier': short_term_config['n_classifier'],
        'n_item': base.shape[0],
        'program_result_dir': program_result_dir,
        'mutual_attribute': short_term_config['mutual_attribute'],
        'result_integrate': short_term_config['result_integrate'],
        'efSearch_l': long_term_config['efSearch']
    }
    result_integrate.integrate(query, gnd, train_model_ins_l, partition_info_l, result_integrate_config)

    save_config_config = {
        'program_result_dir': program_result_dir,
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run
    }
    save_result_config(save_config_config)
