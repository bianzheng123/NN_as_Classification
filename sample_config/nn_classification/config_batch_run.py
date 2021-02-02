import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/small_ds'
    increase_weight_l = [40, 50, 60, 70, 80]
    save_fname_content_m = [
        {
            "type": "knn",
            "build_graph": {
                "increase_weight": 2
            },
            "graph_partition": "parhip"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 16
        config['n_instance'] = 8
        config['train_model']['n_epochs'] = 12
        for increase_weight in increase_weight_l:
            config['specific_fname'] = "increase_weight_%d" % increase_weight
            config['dataset_partition']['build_graph']['increase_weight'] = increase_weight
            with open('%s/%d_%s_%d_increase_weight_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster'], increase_weight),
                      'w') as f:
                json.dump(config, f)
