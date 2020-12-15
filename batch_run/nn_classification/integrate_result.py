import os


def run_nohup(config_dir, fname):
    os.system('nohup python3 integrate_result.py --integrate_result_config_dir %s > ./log/%s.log 2>&1 &' % (
        config_dir, fname))


def run_frontend(config_dir):
    os.system('python3 integrate_result.py --integrate_result_config_dir %s' % config_dir)


if __name__ == '__main__':
    config_sub_dir = '/home/bz/NN_as_Classification/config/integrate_result_3/specific_config/siftsmall/'
    # config_sub_dir = '/home/bz/NN_as_Classification/config/integrate_result_3/specific_config/sift/'
    # config_fname_l = ['siftsmall_1_hnsw', 'siftsmall_1_kmeans_independent', 'siftsmall_1_kmeans_multiple', 'siftsmall_1_knn',
    #                 'siftsmall_8_hnsw', 'siftsmall_8_kmeans_independent', 'siftsmall_8_kmeans_multiple', 'siftsmall_8_knn']
    config_fname_l = ['siftsmall_1_kmeans_multiple']
    # config_fname_l = ['sift_4_hnsw', 'sift_4_kmeans_independent', 'sift_4_kmeans_multiple', 'sift_4_knn']
    # config_fname_l = ['siftsmall_1_kmeans_multiple']
    for fname in config_fname_l:
        dire = config_sub_dir + fname + '.json'
        run_nohup(dire, fname + '_integrate_result')
