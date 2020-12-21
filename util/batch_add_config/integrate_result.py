import json


def write_config(ds_fname, _type, save_base_dir, save_fname_content_l, efSearch_l):
    for fname in save_fname_content_l:
        # config['data_fname'] = 'sift'
        config['data_fname'] = 'siftsmall'
        config['efSearch_l'] = efSearch_l
        config['classifier_fname_l'] = [ds_fname + "_" + fname + _type]
        config['program_fname'] = ds_fname + "_" + fname + _type
        with open('%s/%s.json' % (save_base_dir, fname), 'w') as f:
            json.dump(config, f)


if __name__ == '__main__':
    config_dir = '/home/bianzheng/NN_as_Classification/config/integrate_result/config.json'
    with open(config_dir, 'r') as f:
        config = json.load(f)

    ds_fname = 'siftsmall'
    efSearch_l = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    _type = '_counting_index'
    save_base_dir = '/home/bianzheng/NN_as_Classification/config/integrate_result/counting_index/%s/' % ds_fname
    save_fname_content_l = ['1_kmeans_16', '1_lsh_16', '1_kmeans_multiple_16',
                            '8_kmeans_16', '8_lsh_16', '8_kmeans_multiple_16']

    write_config(ds_fname, _type, save_base_dir, save_fname_content_l, efSearch_l)

    _type = '_nn_classification'
    save_base_dir = '/home/bianzheng/NN_as_Classification/config/integrate_result/nn_classification/%s/' % ds_fname
    save_fname_content_l = ['1_hnsw_16', '1_kmeans_independent_16', '1_kmeans_multiple_16', '1_knn_16',
                            '8_hnsw_16', '8_kmeans_independent_16', '8_kmeans_multiple_16', '8_knn_16']

    write_config(ds_fname, _type, save_base_dir, save_fname_content_l, efSearch_l)
