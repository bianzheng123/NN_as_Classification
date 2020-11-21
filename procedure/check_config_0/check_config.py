'''
输入short_term_config
查看是否各个分类器的配置个数是否对应上
'''


def check_config(config):
    n_classifier = config['n_classifier']
    dataset_partition = config['dataset_partition']
    prepare_train_sample = config['prepare_train_sample']
    train_model = config['train_model']
    if n_classifier != len(dataset_partition) or n_classifier != len(prepare_train_sample) or n_classifier != len(
            train_model):
        raise Exception('分类器的配置个数无法对应')
