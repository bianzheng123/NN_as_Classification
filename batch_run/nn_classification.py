import os


def run_nohup(long_config_dir, short_config_dir, dataset_name, fname):
    os.system('nohup python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type nn_classification > '
              './log/nn_classification/%s_%s.log 2>&1 &' % (
                  long_config_dir, short_config_dir, dataset_name, fname))


def run_frontend(long_config_dir, short_config_dir):
    os.system('python3 run.py --long_term_config_dir %s --short_term_config_dir %s --type nn_classification' % (
        long_config_dir, short_config_dir))


if __name__ == '__main__':
    ds_fname = 'sift'
    base_config_dir = '/home/bianzheng/NN_as_Classification/config/nn_classification/%s/' % ds_fname
    long_config_dir = base_config_dir + 'long_term_config.json'

    # short_config_fname_arr = ['1_hnsw_16', '1_kmeans_independent_16', '1_kmeans_multiple_16', '1_knn_16']
    # short_config_fname_arr = ['1_knn_parhip_16']
    # short_config_fname_arr = ['1_knn_16']
    # short_config_fname_arr = ['1_kmeans_multiple_256', '1_knn_256',
    #                           '4_kmeans_multiple_256', '4_knn_256']
    short_config_fname_arr = ['4_kmeans_multiple_256', '4_knn_parhip_256']
    # short_config_fname_arr = ['4_kmeans_multiple_256', '4_knn_256']
    # short_config_fname_arr = ['8_kmeans_multiple_16']
    for tmp_fname in short_config_fname_arr:
        short_config_dir = base_config_dir + tmp_fname + '.json'
        run_nohup(long_config_dir, short_config_dir, ds_fname, tmp_fname)
        # run_frontend(long_config_dir, short_config_dir)
