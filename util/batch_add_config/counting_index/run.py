import json

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/counting_index/run_2/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/bz/NN_as_Classification/config/counting_index/run_2/specific_config/siftsmall'
    save_fname_content_m = {
        'siftsmall_1_kmeans_multiple': {
            "n_instance": 1,
            "type": "kmeans_multiple",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'siftsmall_1_kmeans': {
            "n_instance": 1,
            "type": "kmeans",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'siftsmall_1_lsh': {
            "n_instance": 1,
            "type": "e2lsh",
            "dataset_partition": {
                "r": 1,
                "a_sigma": 1,
                "a_miu": 0,
                "n_mod": 4
            }
        },
        'siftsmall_8_kmeans_multiple': {
            "n_instance": 8,
            "type": "kmeans_multiple",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'siftsmall_8_kmeans': {
            "n_instance": 8,
            "type": "kmeans",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        'siftsmall_8_lsh': {
            "n_instance": 8,
            "type": "e2lsh",
            "dataset_partition": {
                "r": 1,
                "a_sigma": 1,
                "a_miu": 0,
                "n_mod": 4
            }
        }
    }
    for fname in save_fname_content_m:
        config['independent_config'] = [save_fname_content_m[fname]]
        config['program_fname'] = fname
        config['n_cluster'] = 16
        with open('%s/%s.json' % (save_base_dir, fname), 'w') as f:
            json.dump(config, f)
