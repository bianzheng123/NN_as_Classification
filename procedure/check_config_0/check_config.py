'''
输入short_term_config
查看是否各个分类器的配置个数是否对应上
'''


def check_config(config):
    mutual_attribute = config['mutual_attribute']
    dataset_partition = config['dataset_partition']
    prepare_train_sample = config['prepare_train_sample']
    train_model = config['train_model']
    if len(mutual_attribute) != len(dataset_partition) or len(mutual_attribute) != len(prepare_train_sample) or len(
            mutual_attribute) != len(train_model):
        raise Exception('分类器的配置个数无法对应')
    # 新增一个n_total_classifier
    n_total_classifier = 0
    for tmp_config in mutual_attribute:
        n_total_classifier += tmp_config['n_instance']
    config['n_total_classifier'] = n_total_classifier
    config['n_config_entity'] = len(mutual_attribute)
