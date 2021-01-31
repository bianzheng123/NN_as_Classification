import _init_paths

import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/big_ds'
    for k_graph in [20, 30, 40, 50, 60]:
        config['dataset_partition'] = {
            "type": "knn",
            "build_graph": {
                "k_graph": k_graph,
                "increase_weight": 2
            },
            "graph_partition": "parhip"
        }
        config['n_cluster'] = 256
        config['n_instance'] = 4
        config['specific_fname'] = 'k_graph_%d' % k_graph
        config['train_model']['n_epochs'] = 12
        with open('%s/%d_%s_%d_k_graph_%d.json' % (
                save_base_dir, config['n_instance'], config['dataset_partition']['type'], config['n_cluster'], k_graph),
                  'w') as f:
            json.dump(config, f)
