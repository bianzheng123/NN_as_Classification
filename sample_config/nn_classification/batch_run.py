import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/big_ds'
    save_fname_content_m = [
        {
            "type": "knn_random_projection",
            "build_graph": {
            },
            "graph_partition": "parhip"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 256
        config['n_instance'] = 4
        with open('%s/%d_%s_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster']),
                  'w') as f:
            json.dump(config, f)
