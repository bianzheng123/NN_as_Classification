import json

if __name__ == '__main__':
    with open('/home/zhengbian/NN_as_Classification/sample_config/nn_classification/short_term_config.json', 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/zhengbian/NN_as_Classification/config/nn_classification/small_ds'
    save_fname_content_m = [
        {
            "type": "hnsw",
            "build_graph": {},
            "graph_partition": "kaffpa"
        },
        {
            "type": "kmeans_independent"
        },
        {
            "type": "kmeans_multiple"
        },
        {
            "type": "knn",
            "build_graph": {
            },
            "graph_partition": "parhip"
        },
        {
            "type": "random_hash"
        },
        {
            "type": "lsh"
        }
    ]
    for tmp_config in save_fname_content_m:
        config['dataset_partition'] = tmp_config
        config['n_cluster'] = 16
        config['n_instance'] = 1
        with open('%s/%d_%s_%d.json' % (save_base_dir, config['n_instance'], tmp_config['type'], config['n_cluster']),
                  'w') as f:
            json.dump(config, f)
