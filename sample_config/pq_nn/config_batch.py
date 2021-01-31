import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/pq_nn/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/pq_nn/small_ds'
    save_fname_content_m = [
        {
            "type": "knn",
            "build_graph": {},
            "graph_partition": "parhip"
        }
    ]
    for tmp_config in save_fname_content_m:
        for n_hidden in [128, 256, 512]:
            config['dataset_partition'] = tmp_config
            config['n_cluster'] = 16
            config['n_instance'] = 8
            config['train_model']['n_hidden'] = n_hidden
            config['train_model']['n_epochs'] = 12
            config['specific_fname'] = "n_hidden_%d" % n_hidden
            with open('%s/%d_%s_%d_n_hidden_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster'], n_hidden),
                      'w') as f:
                json.dump(config, f)
