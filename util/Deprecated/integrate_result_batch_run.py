import os


def run_nohup(config_dir, fname, _type):
    os.system(
        'nohup python3 -u integrate_result.py --integrate_result_config_dir %s > ./log/%s/integrate_result_%s.log '
        '2>&1 &' % (
            config_dir, _type, fname))


def run_frontend(config_dir, _type):
    os.system('python3 integrate_result.py --integrate_result_config_dir %s' % config_dir)


def run(sub_dir, fname_l, _type):
    for fname in fname_l:
        dire = sub_dir + fname + '.json'
        run_nohup(dire, fname, _type)
        # run_frontend(dire, _type)


if __name__ == '__main__':
    dataset_fname = "siftsmall"

    config_sub_dir = '/home/bianzheng/NN_as_Classification/config/integrate_result/counting_index/%s/' % dataset_fname
    config_fname_l = ['1_kmeans_16', '1_kmeans_multiple_16', '1_lsh_16',
                      '8_kmeans_16', '8_kmeans_multiple_16', '8_lsh_16']
    run(config_sub_dir, config_fname_l, 'counting_index')

    config_sub_dir = '/home/bianzheng/NN_as_Classification/config/integrate_result/nn_classification/%s/' % dataset_fname
    config_fname_l = ['1_hnsw_16', '1_kmeans_independent_16', '1_kmeans_multiple_16', '1_knn_16',
                      '8_hnsw_16', '8_kmeans_independent_16', '8_kmeans_multiple_16', '8_knn_16']

    run(config_sub_dir, config_fname_l, 'nn_classification')
