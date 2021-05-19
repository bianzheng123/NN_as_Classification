import json
import math

if __name__ == '__main__':
    config_dir = '/home/zhengbian/NN_as_Classification/sample_config/counting_index/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/counting_index/small_ds'
    save_fname_content_m = [
        {
            "type": "kmeans_multiple",
            "max_iter": 40
        }
    ]

    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        n_cluster = 16
        n_instance = 1
        config['n_cluster'] = n_cluster
        config['n_instance'] = n_instance
        if tmp_config['type'] == 'e2lsh':
            config['n_cluster'] = int(math.sqrt(n_cluster))
            config['n_instance'] = n_instance * 2
        with open('%s/%d_%s_%d.json' % (save_base_dir, n_instance, tmp_config['type'], n_cluster), 'w') as f:
            json.dump(config, f)
