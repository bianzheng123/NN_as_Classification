import _init_paths

import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/small_ds'
    for label_k in [20, 40, 60, 80, 100]:
        config['dataset_partition'] = {
            "type": "knn",
            "build_graph": {
                "increase_weight": 2
            },
            "graph_partition": "parhip"
        }
        config['n_cluster'] = 16
        config['n_instance'] = 1
        config['prepare_train_sample']['label_k'] = label_k
        config['specific_fname'] = 'label_k_%d' % label_k
        config['train_model']['n_epochs'] = 12
        with open('%s/%d_%s_%d_label_k_%d.json' % (
                save_base_dir, config['n_instance'], config['dataset_partition']['type'], config['n_cluster'], label_k),
                  'w') as f:
            json.dump(config, f)
