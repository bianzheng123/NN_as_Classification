import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/big_ds'
    specific_fname = 'model'
    model_name = ['std_nn', 'res_net', 'one_block_2048_dim']  # cnn two_block_8192_dim_no_bn_dropout
    save_fname_content_m = [
        {
            "type": "knn",
            "build_graph": {
            },
            "graph_partition": "parhip"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 256
        config['n_instance'] = 1
        for para in model_name:
            config['specific_fname'] = "{}_{}".format(specific_fname, para)
            config['train_model']['model_name'] = para
            dire = '{}/{}_{}_{}_{}_{}.json'.format(save_base_dir, config['n_instance'], tmp_config['type'],
                                                   config['n_cluster'], specific_fname, para)
            with open(dire,
                      'w') as f:
                json.dump(config, f)
