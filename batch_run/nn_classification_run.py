import os


def run_nohup(long_config_dir, short_config_dir, dataset_name, fname):
    os.system('nohup python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type nn_classification > '
              './log/nn_classification/%s_%s.log 2>&1 &' % (
                  long_config_dir, short_config_dir, dataset_name, fname))


def run_frontend(long_config_dir, short_config_dir):
    os.system('python3 run.py --long_term_config_dir %s --short_term_config_dir %s --type nn_classification' % (
        long_config_dir, short_config_dir))


if __name__ == '__main__':
    ds_fname = 'siftsmall'
    base_config_dir = '/home/bianzheng/NN_as_Classification/config/nn_classification/%s/' % ds_fname
    # base_config_dir = '/home/bianzheng/NN_as_Classification/config/nn_classification/sift/'
    long_config_dir = base_config_dir + 'long_term_config.json'

    short_config_fname_arr = ['1_hnsw_16', '1_kmeans_independent_16', '1_kmeans_multiple_16', '1_knn_16',
                              '8_hnsw_16', '8_kmeans_independent_16', '8_kmeans_multiple_16', '8_knn_16']
    # short_config_fname_arr = ['1_kmeans_independent_16']
    # short_config_fname_arr = ['sift_1_hnsw', 'sift_1_kmeans_independent', 'sift_1_kmeans_multiple', 'sift_1_knn',
    #                           'sift_2_hnsw', 'sift_2_kmeans_independent', 'sift_2_kmeans_multiple', 'sift_2_knn',
    #                           'sift_4_hnsw', 'sift_4_kmeans_independent', 'sift_4_kmeans_multiple', 'sift_4_knn']
    for tmp_fname in short_config_fname_arr:
        short_config_dir = base_config_dir + tmp_fname + '.json'
        run_nohup(long_config_dir, short_config_dir, ds_fname, tmp_fname)
        # run_frontend(long_config_dir, short_config_dir)
