import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/small_ds'
    config_l = [10**i for i in range(15)]
    save_fname_content_m = [
        {
            "type": "knn",
            "build_graph": {
                "increase_weight": 60
            },
            "graph_partition": "parhip"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 16
        config['n_instance'] = 8
        for para in config_l:
            config['specific_fname'] = "increase_weight_%d" % para
            config['dataset_partition']['build_graph']['increase_weight'] = para
            with open('%s/%d_%s_%d_increase_weight_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster'], para),
                      'w') as f:
                json.dump(config, f)
