import os


def run_nohup(long_config_dir, short_config_dir, fname):
    os.system('nohup python3 run.py --long_term_config_dir %s --short_term_config_dir %s > %s.log 2>&1 &' % (
        long_config_dir, short_config_dir, fname))


def run_frontend(long_config_dir, short_config_dir):
    os.system('python3 run.py --long_term_config_dir %s --short_term_config_dir %s' % (
        long_config_dir, short_config_dir))


if __name__ == '__main__':
    base_config_dir = '/home/bz/NN_as_Classification/config/run_2/specific_config/siftsmall/'
    # base_config_dir = '/home/bz/NN_as_Classification/config/run_2/specific_config/sift/'
    long_config_dir = base_config_dir + 'long_term_config.json'

    # short_config_fname_arr = ['siftsmall_2_hnsw', 'siftsmall_2_kmeans_independent',
    #                           'siftsmall_2_kmeans_multiple_batch', 'siftsmall_2_knn']
    short_config_fname_arr = ['siftsmall_2_kmeans_multiple_batch']
    # short_config_fname_arr = ['sift_2_hnsw', 'sift_2_kmeans_independent', 'sift_2_kmeans_multiple_batch',
    #                           'sift_2_knn']
    for tmp_fname in short_config_fname_arr:
        short_config_dir = base_config_dir + tmp_fname + '.json'
        # run_nohup(long_config_dir, short_config_dir, tmp_fname)
        run_frontend(long_config_dir, short_config_dir)
