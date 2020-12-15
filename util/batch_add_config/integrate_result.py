import json

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/integrate_result/config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)

    _type = '_counting_index'
    save_base_dir = '/home/bz/NN_as_Classification/config/integrate_result/counting_index/siftsmall'
    save_fname_content_l = ['siftsmall_1_kmeans_16', 'siftsmall_1_lsh_16', 'siftsmall_1_kmeans_multiple_16']

    # _type = '_nn_classification'
    # save_base_dir = '/home/bz/NN_as_Classification/config/integrate_result/nn_classification/siftsmall'
    # save_fname_content_l = ['siftsmall_1_kmeans_multiple_16', 'siftsmall_1_hnsw_16', 'siftsmall_1_kmeans_independent_16',
    #                         'siftsmall_1_knn_16']

    # efSearch_l = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    efSearch_l = [300, 500, 700, 900, 1100, 1300, 1500, 1700, 1900, 2100]
    # for i in range(len(efSearch_l)):
    #     efSearch_l[i] *= 10000

    for fname in save_fname_content_l:
        # config['data_fname'] = 'sift'
        config['data_fname'] = 'siftsmall'
        config['efSearch_l'] = efSearch_l
        config['classifier_fname_l'] = [fname + _type]
        config['program_fname'] = fname + _type
        with open('%s/%s.json' % (save_base_dir, fname), 'w') as f:
            json.dump(config, f)
