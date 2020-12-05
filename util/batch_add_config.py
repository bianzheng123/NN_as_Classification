import json

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/run_2/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/bz/NN_as_Classification/config/run_2/specific_config/sift'
    save_fname_content_m = {
        'sift_1_knn': {
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
                    "time_limit": 180
                }
            }
        },
        'sift_1_hnsw': {
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
                    "time_limit": 180
                }
            }
        },
        'sift_1_kmeans_independent': {
            "n_instance": 1,
            "type": "kmeans",
            "specific_type": "independent",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'sift_1_kmeans_multiple_batch': {
            "n_instance": 1,
            "type": "kmeans",
            "specific_type": "multiple_batch",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'sift_8_knn': {
            "n_instance": 8,
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
                    "time_limit": 180
                }
            }
        },
        'sift_8_hnsw': {
            "n_instance": 8,
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
                    "time_limit": 180
                }
            }
        },
        'sift_8_kmeans_independent': {
            "n_instance": 8,
            "type": "kmeans",
            "specific_type": "independent",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'sift_8_kmeans_multiple_batch': {
            "n_instance": 8,
            "type": "kmeans",
            "specific_type": "multiple_batch",
            "dataset_partition": {
                "max_iter": 40
            }
        }
    }
    for fname in save_fname_content_m:
        config['independent_config'] = [save_fname_content_m[fname]]
        config['program_fname'] = fname
        with open('%s/%s.json' % (save_base_dir, fname), 'w') as f:
            json.dump(config, f)
