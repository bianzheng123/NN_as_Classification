import json

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/nn_classification/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/bz/NN_as_Classification/config/nn_classification/siftsmall'
    save_fname_content_m = {
        'siftsmall_1_knn': {
            "n_instance": 1,
            "type": "learn_on_graph",
            "specific_type": "knn",
            "dataset_partition": {
                "build_graph": {
                    "type": "knn",
                    "k_graph": 10
                },
                "graph_partition": {
                    "type": "kahip",
                    "preconfiguration": "eco",
                    "time_limit": 300
                }
            }
        },
        'siftsmall_1_hnsw': {
            "n_instance": 1,
            "type": "learn_on_graph",
            "specific_type": "hnsw",
            "dataset_partition": {
                "build_graph": {
                    "type": "hnsw",
                    "k_graph": 10
                },
                "graph_partition": {
                    "type": "kahip",
                    "preconfiguration": "eco",
                    "time_limit": 300
                }
            }
        },
        'siftsmall_1_kmeans_independent': {
            "n_instance": 1,
            "type": "kmeans",
            "specific_type": "independent",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'siftsmall_1_kmeans_multiple': {
            "n_instance": 1,
            "type": "kmeans",
            "specific_type": "multiple",
            "dataset_partition": {
                "max_iter": 40
            }
        }
    }
    # save_fname_content_m = {
    #     'siftsmall_2_kmeans_multiple': {
    #         "n_instance": 2,
    #         "type": "kmeans",
    #         "specific_type": "multiple",
    #         "dataset_partition": {
    #             "max_iter": 40
    #         }
    #     }
    # }
    for fname in save_fname_content_m:
        config['independent_config'] = [save_fname_content_m[fname]]
        config['n_cluster'] = 16
        config['program_fname'] = fname + "_" + str(config['n_cluster']) + "_nn_classification"
        with open('%s/%s_%d.json' % (save_base_dir, fname, config['n_cluster']), 'w') as f:
            json.dump(config, f)
