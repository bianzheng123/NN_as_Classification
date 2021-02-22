import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/small_ds'
    config_l = [1, 2, 3, 4, 5]
    save_fname_content_m = [
        {
            "type": "partition_knn",
            "build_graph": {
            },
            "graph_partition": "parhip"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 16
        config['n_instance'] = 4
        for para in config_l:
            config['specific_fname'] = "partition_depth_%d" % para
            config['dataset_partition']['build_graph']['partition_depth'] = para
            with open('%s/%d_%s_%d_partition_depth_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster'], para),
                      'w') as f:
                json.dump(config, f)
