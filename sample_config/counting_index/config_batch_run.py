import json
import math

if __name__ == '__main__':
    config_dir = '/home/zhengbian/NN_as_Classification/sample_config/counting_index/short_term_config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    dataset_name = 'siftsmall'
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/counting_index/small_ds'
    a_sigma_l = [1, 2, 4, 8, 16, 32, 64]
    save_fname_content_m = [
        {
            "type": "e2lsh",
            "r": 1,
            "a_sigma": 1,
            "a_miu": 0
        }
    ]

    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        for sigma in a_sigma_l:
            n_cluster = 16
            n_instance = 1
            config['n_cluster'] = n_cluster
            config['dataset_partition']['a_sigma'] = sigma
            config['specific_fname'] = 'a_sigma_%d' % sigma
            config['n_instance'] = n_instance
            if tmp_config['type'] == 'e2lsh':
                config['n_cluster'] = int(math.sqrt(n_cluster))
                config['n_instance'] = n_instance * 2
            with open('%s/%d_%s_%d_a_sigma_%d.json' % (save_base_dir, n_instance, tmp_config['type'], n_cluster, sigma), 'w') as f:
                json.dump(config, f)
