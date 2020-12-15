import os


def run_nohup(config_dir, fname, _type):
    os.system(
        'nohup python3 integrate_result.py --integrate_result_config_dir %s > ./log/%s_%s.log 2>&1 &' % (
            config_dir, fname, _type))


def run_frontend(config_dir, _type):
    os.system('python3 integrate_result.py --integrate_result_config_dir %s' % config_dir)


def run(sub_dir, fname_l, _type):
    for fname in fname_l:
        dire = sub_dir + fname + '.json'
        # run_nohup(dire, fname + '_integrate_result')
        run_frontend(dire, _type)


if __name__ == '__main__':
    config_sub_dir = '/home/bz/NN_as_Classification/config/integrate_result/counting_index/siftsmall/'
    config_fname_l = ['siftsmall_1_kmeans_16', 'siftsmall_1_kmeans_multiple_16', 'siftsmall_1_lsh_16']

    run(config_sub_dir, config_fname_l, 'counting_index')

    config_sub_dir = '/home/bz/NN_as_Classification/config/integrate_result/nn_classification/siftsmall/'
    config_fname_l = ['siftsmall_1_kmeans_multiple_16', 'siftsmall_1_knn_16'
                      # , 'siftsmall_1_hnsw_16','siftsmall_1_kmeans_independent_16'
                      ]

    run(config_sub_dir, config_fname_l, 'nn_classification')
