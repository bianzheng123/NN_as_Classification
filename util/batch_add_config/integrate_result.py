import json

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/integrate_result_3/config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    save_base_dir = '/home/bz/NN_as_Classification/config/integrate_result_3/specific_config/sift'
    save_fname_content_l = ['sift_4_knn', 'sift_4_hnsw', 'sift_4_kmeans_independent',
                            'sift_4_kmeans_multiple']

    efSearch_l = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in range(len(efSearch_l)):
        efSearch_l[i] *= 10000

    for fname in save_fname_content_l:
        config['data_fname'] = 'sift'
        config['efSearch_l'] = efSearch_l
        config['classifier_fname_l'] = [fname]
        config['program_fname'] = fname
        with open('%s/%s.json' % (save_base_dir, fname), 'w') as f:
            json.dump(config, f)
