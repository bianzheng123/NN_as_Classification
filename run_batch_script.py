import os

if __name__ == '__main__':
    base_config_dir = '/home/bz/NN_as_Classification/config/run_2/specific_config/sift/'
    long_config_dir = base_config_dir + 'long_term_config.json'

    # short_config_fname_arr = ['siftsmall_2_hnsw', 'siftsmall_2_kmeans_independent',
    #                           'siftsmall_2_kmeans_multiple_batch', 'siftsmall_2_knn']
    short_config_fname_arr = ['sift_1_hnsw', 'sift_1_kmeans_independent',
                              'sift_1_kmeans_multiple_batch', 'sift_1_knn', 'sift_8_hnsw', 'sift_8_kmeans_independent',
                              'sift_8_kmeans_multiple_batch', 'sift_8_knn']
    for tmp_fname in short_config_fname_arr:
        short_config_dir = base_config_dir + tmp_fname + '.json'
        os.system('nohup python3 run.py --long_term_config_dir %s --short_term_config_dir %s > %s.log 2>&1 &' % (
            long_config_dir, short_config_dir, tmp_fname))
