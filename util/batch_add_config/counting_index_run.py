import json

if __name__ == '__main__':
    config_dir = '/home/zhengbian/NN_as_Classification/config/counting_index/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    dataset_name = 'siftsmall'
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/counting_index/%s' % dataset_name
    save_fname_content_m = {
        '1_kmeans_multiple': {
            "n_instance": 1,
            "type": "kmeans_multiple",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        '1_kmeans_independent': {
            "n_instance": 1,
            "type": "kmeans_independent",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        '1_lsh': {
            "n_instance": 2,
            "type": "e2lsh",
            "dataset_partition": {
                "r": 1,
                "a_sigma": 1,
                "a_miu": 0
            }
        },

        '8_kmeans_multiple': {
            "n_instance": 8,
            "type": "kmeans_multiple",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        '8_kmeans_independent': {
            "n_instance": 8,
            "type": "kmeans_independent",
            "dataset_partition": {
                "max_iter": 40
            }
        },
        '8_lsh': {
            "n_instance": 16,
            "type": "e2lsh",
            "dataset_partition": {
                "r": 1,
                "a_sigma": 1,
                "a_miu": 0
            }
        }
    }
    for fname in save_fname_content_m:
        config['independent_config'] = [save_fname_content_m[fname]]
        n_cluster = 16
        config['n_cluster'] = n_cluster
        if fname.split('_')[-1] == 'lsh':
            config['n_cluster'] = 4
        config['program_fname'] = '%s_%d_count_%s' % (dataset_name, n_cluster, fname)
        with open('%s/%s_%d.json' % (save_base_dir, fname, n_cluster), 'w') as f:
            json.dump(config, f)
