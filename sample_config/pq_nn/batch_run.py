import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/pq_nn/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/pq_nn/small_ds'
    save_fname_content_m = [
        {
            "type": "lsh"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 16
        config['n_instance'] = 1
        config['train_model']['n_epochs'] = 12
        with open('%s/%d_%s_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster']),
                  'w') as f:
            json.dump(config, f)
