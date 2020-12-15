import json

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/counting_index/integrate_result_3/config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)
    # save_base_dir = '/home/bz/NN_as_Classification/config/counting_index/integrate_result_3/specific_config/sift'
    save_base_dir = '/home/bz/NN_as_Classification/config/counting_index/integrate_result_3/specific_config/siftsmall'
    # save_fname_content_l = ['sift_4_knn', 'sift_4_hnsw', 'sift_4_kmeans_independent',
    #                         'sift_4_kmeans_multiple']
    save_fname_content_l = ['siftsmall_1_kmeans', 'siftsmall_1_lsh', 'siftsmall_1_kmeans_multiple',
                            'siftsmall_8_kmeans', 'siftsmall_8_lsh', 'siftsmall_8_kmeans_multiple']

    # efSearch_l = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    efSearch_l = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100]
    # for i in range(len(efSearch_l)):
    #     efSearch_l[i] *= 10000

    for fname in save_fname_content_l:
        # config['data_fname'] = 'sift'
        config['data_fname'] = 'siftsmall'
        config['efSearch_l'] = efSearch_l
        config['classifier_fname_l'] = [fname]
        config['program_fname'] = fname
        with open('%s/%s.json' % (save_base_dir, fname), 'w') as f:
            json.dump(config, f)
