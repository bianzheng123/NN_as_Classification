import os


def run_nohup(long_config_dir, short_config_dir, fname):
    os.system('nohup python3 run.py --long_term_config_dir %s --short_term_config_dir %s --type nn_classification > '
              './log/%s.log 2>&1 &' % (
                  long_config_dir, short_config_dir, fname))


def run_frontend(long_config_dir, short_config_dir):
    os.system('python3 run.py --long_term_config_dir %s --short_term_config_dir %s --type nn_classification' % (
        long_config_dir, short_config_dir))


if __name__ == '__main__':
    base_config_dir = '/home/bz/NN_as_Classification/config/nn_classification/siftsmall/'
    # base_config_dir = '/home/bz/NN_as_Classification/config/nn_classification/run_2/specific_config/sift/'
    long_config_dir = base_config_dir + 'long_term_config.json'

    # short_config_fname_arr = ['siftsmall_1_kmeans_multiple', 'siftsmall_1_hnsw', 'siftsmall_1_kmeans_independent',
    #                           'siftsmall_1_knn',
    #                           'siftsmall_8_kmeans_multiple', 'siftsmall_8_hnsw', 'siftsmall_8_kmeans_independent',
    #                           'siftsmall_8_knn']
    short_config_fname_arr = ['siftsmall_1_kmeans_multiple_16', #'siftsmall_1_hnsw_16',
                              #'siftsmall_1_kmeans_independent_16',
                              'siftsmall_1_knn_16']
    # short_config_fname_arr = ['siftsmall_2_kmeans_multiple']
    # short_config_fname_arr = ['sift_1_hnsw', 'sift_1_kmeans_independent', 'sift_1_kmeans_multiple', 'sift_1_knn',
    #                           'sift_2_hnsw', 'sift_2_kmeans_independent', 'sift_2_kmeans_multiple', 'sift_2_knn',
    #                           'sift_4_hnsw', 'sift_4_kmeans_independent', 'sift_4_kmeans_multiple', 'sift_4_knn']
    for tmp_fname in short_config_fname_arr:
        short_config_dir = base_config_dir + tmp_fname + '.json'
        # run_nohup(long_config_dir, short_config_dir, tmp_fname)
        run_frontend(long_config_dir, short_config_dir)
