import json

if __name__ == '__main__':
    config_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    dataset_name = 'siftsmall'
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/%s' % dataset_name
    save_fname_content_m = {
        "8_knn": {
            "n_instance": 8,
            "type": "learn_on_graph",
            "specific_type": "knn",
            "dataset_partition": {
                "build_graph": {
                    "type": "knn",
                    "k_graph": 10,
                    "increase_weight": 10
                },
                "graph_partition": {
                    "type": "parhip",
                    "preconfiguration": "fastsocial"
                }
            }
        },
        "8_hnsw": {
            "n_instance": 8,
            "type": "learn_on_graph",
            "specific_type": "hnsw",
            "dataset_partition": {
                "build_graph": {
                    "type": "hnsw",
                    "k_graph": 10
                },
                "graph_partition": {
                    "type": "parhip",
                    "preconfiguration": "fastsocial"
                }
            }
        },
        "8_kmeans_independent": {
            "n_instance": 8,
            "type": "kmeans",
            "specific_type": "independent",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        "8_kmeans_multiple": {
            "n_instance": 8,
            "type": "kmeans",
            "specific_type": "multiple",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        "8_random": {
            "n_instance": 8,
            "type": "random_hash",
            "dataset_partition": {
            }
        }
    }
    for fname in save_fname_content_m:
        config['independent_config'] = [save_fname_content_m[fname]]
        config['n_cluster'] = 16
        config['train_model']['n_epochs'] = 12
        config['program_fname'] = '%s_%d_nn_%s' % (dataset_name, config['n_cluster'], fname)
        with open('%s/%s_%d.json' % (save_base_dir, fname, config['n_cluster']), 'w') as f:
            json.dump(config, f)
