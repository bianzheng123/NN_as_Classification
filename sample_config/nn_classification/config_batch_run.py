import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/big_ds'
    specific_fname = 'graph_partition'
    # para_l = ['two_block_512_dim', 'two_block_1024_dim', 'one_block_2048_dim', 'one_block_512_dim',
    #           'two_block_512_dim_no_bn_dropout', 'res_net']  # cnn two_block_8192_dim_no_bn_dropout
    para_l = ['kaffpa', 'parhip']  # cnn two_block_8192_dim_no_bn_dropout
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
        for para in para_l:
            config['specific_fname'] = "{}_{}".format(specific_fname, para)
            # config['dataset_partition']['build_graph']['partition_iter'] = para
            config['dataset_partition']['graph_partition'] = para
            dire = '{}/{}_{}_{}_{}_{}.json'.format(save_base_dir, config['n_instance'], tmp_config['type'],
                                                   config['n_cluster'], specific_fname, para)
            with open(dire,
                      'w') as f:
                json.dump(config, f)
