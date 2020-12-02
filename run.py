from util.numpy import load_data
from procedure.init_0 import partition_preprocess
from procedure.dataset_partition_1 import dataset_partition
from procedure.prepare_train_sample_2 import prepare_train_sample
from procedure.train_eval_model_3 import train_eval_model
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


def delete_origin_dir(project_dir, project_fname):
    train_para_dir = '%s/train_para/%s' % (project_dir, project_fname)
    delete_dir_if_exist(train_para_dir)
    result_dir = '%s/result/%s' % (project_dir, project_fname)
    delete_dir_if_exist(result_dir)


def save_result_config(config):
    save_dir = '%s/config' % config['program_result_dir']
    os.system('mkdir %s' % save_dir)

    long_config = config['long_term_config']
    short_config = config['short_term_config']
    short_config_before_run = config['short_term_config_before_run']
    intermediate_result = config['intermediate_result']

    result_integrate.save_json(save_dir, 'long_term_config.json', long_config)
    result_integrate.save_json(save_dir, 'short_term_config.json', short_config)
    result_integrate.save_json(save_dir, 'short_term_config_before_run.json', short_config_before_run)
    result_integrate.save_json(config['program_result_dir'], 'intermediate_result.json', intermediate_result)
    print('save program: %s' % config['program_fname'])


def train_eval(long_term_config_dir, short_term_config_dir):
    with open(long_term_config_dir, 'r') as f:
        long_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config = json.load(f)
    with open(short_term_config_dir, 'r') as f:
        short_term_config_before_run = json.load(f)

    delete_origin_dir(long_term_config['project_dir'], short_term_config['program_fname'])

    # 加载数据
    data_dir = '%s/data/%s_%d' % (
        long_term_config['project_dir'], long_term_config['data_fname'], long_term_config['k'])
    load_data_config = {
        'data_dir': data_dir
    }
    base, query, learn, gnd = load_data.load_data_npy(load_data_config)

    # 将代码变成每一个分类器都单独运行，而不是批量执行某一个步骤
    # 执行切分的不可并行化结果
    program_train_para_dir = '%s/train_para/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    partition_preprocess_config = {
        "independent_config": short_term_config['independent_config'],
        'n_cluster': short_term_config['n_cluster'],
        'kahip_dir': long_term_config['kahip_dir'],
        "program_train_para_dir": program_train_para_dir,
    }
    model_l = partition_preprocess.preprocess(base, partition_preprocess_config)

    result_l = []
    intermediate_para_l = []
    # for 里面的东西并行
    for model in model_l:
        partition_info, model_info = dataset_partition.partition(base, model)
        partition_intermediate = model_info[1]

        prepare_train_config = short_term_config['prepare_train_sample']
        prepare_train_config['n_cluster'] = short_term_config['n_cluster']
        prepare_train_config['program_train_para_dir'] = program_train_para_dir
        prepare_train_config['classifier_number'] = model_info[0]["classifier_number"]
        prepare_train_config['entity_number'] = model_info[0]["entity_number"]
        # 准备训练的代码将要完成
        trainset, prepare_train_intermediate = prepare_train_sample.prepare_train(base, partition_info,
                                                                                  prepare_train_config)

        train_model_config = short_term_config['train_model']
        train_model_config['n_cluster'] = short_term_config['n_cluster']
        train_model_config['classifier_number'] = model_info[0]['classifier_number']
        train_model_config['entity_number'] = model_info[0]['entity_number']
        train_model_config['program_train_para_dir'] = program_train_para_dir

        label_map = partition_info[1]
        tmp_result, train_eval_intermediate = train_eval_model.train_eval_model(base, query, label_map, trainset,
                                                                                train_model_config)

        intermediate = {
            "ins_id": '%d_%d' % (model_info[0]["entity_number"], model_info[0]["classifier_number"]),
            'dataset_partition': partition_intermediate,
            'prepare_train_sample': prepare_train_intermediate,
            'train_eval_model': train_eval_intermediate
        }
        intermediate_para_l.append(intermediate)
        result_l.append(tmp_result)

    program_result_dir = '%s/result/%s' % (long_term_config['project_dir'], short_term_config['program_fname'])
    result_integrate_config = {
        'k': long_term_config['k'],
        'program_result_dir': program_result_dir,
        'result_integrate': short_term_config['result_integrate'],
        'efSearch_l': long_term_config['efSearch']
    }
    # 结果整合与中间结果分开
    result_integrate.integrate(result_l, gnd, result_integrate_config)

    save_config_config = {
        'program_fname': short_term_config['program_fname'],
        'program_result_dir': program_result_dir,
        'long_term_config': long_term_config,
        'short_term_config': short_term_config,
        'short_term_config_before_run': short_term_config_before_run,
        'intermediate_result': intermediate_para_l
    }
    save_result_config(save_config_config)


if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/run/'
    long_config_dir = config_dir + 'long_term_config.json'
    short_config_dir = config_dir + 'short_term_config.json'
    train_eval(long_config_dir, short_config_dir)
