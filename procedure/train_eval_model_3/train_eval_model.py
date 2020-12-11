from procedure.train_eval_model_3 import neural_network
from procedure.result_integrate_4 import result_integrate
import numpy as np
from util import dir_io


def train_eval_model(base, query, trainset, config):
    save_dir = '%s/Classifier_%d_%d' % (
        config['program_train_para_dir'], config['entity_number'], config['classifier_number'])
    config['save_dir'] = save_dir
    train_model_ins = train_model_factory(config)
    # if use the learn dataset, add the learn variable in here
    train_model_ins.train(base, trainset)
    eval_result, intermediate_config = train_model_ins.eval(query)
    # train_model_ins.save()
    return eval_result, intermediate_config


'''
count the total score_table from all the score table in each classifier
'''


def integrate_save_score_table(cluster_score_l, label_map_l, config):
    shape = cluster_score_l[0].shape

    total_score_table_dir = '%s/total_score_table.txt' % config['program_train_para_dir']
    dir_io.save_file(total_score_table_dir)
    f_ptr = open(total_score_table_dir, "w")

    for i in range(shape[0]):
        result_score_single_query = np.zeros(shape=config['n_item'], dtype=np.float32)
        for k, tmp_cluster_score in enumerate(cluster_score_l, 0):
            label_map = label_map_l[k]
            for j in range(shape[1]):
                score_item_idx_l = label_map[j]
                result_score_single_query[score_item_idx_l] += tmp_cluster_score[i][j].item()
        str_line = ' '.join('%.3f' % score for score in result_score_single_query)
        f_ptr.write(str_line + '\n')

    f_ptr.close()

    print('save score table success')


def train_model_factory(config):
    _type = config['type']
    if _type == 'neural_network':
        return neural_network.NeuralNetwork(config)
    raise Exception('do not support the type of training data')
