import os
from util import send_email


def run_nohup(long_config_dir, short_config_dir, dataset_name, fname, _type):
    os.system('nohup python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type %s > '
              './log/%s/%s_%s.log 2>&1 &' % (
                  long_config_dir, short_config_dir, _type, _type, dataset_name, fname))


def run_frontend(long_config_dir, short_config_dir, _type):
    os.system('python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type %s' % (
        long_config_dir, short_config_dir, _type))


'''
nn_classification
short_config_fname_arr = ['1_hnsw_16', '1_kmeans_independent_16', '1_kmeans_multiple_16', '1_knn_16',
                          '1_lsh_16', '1_random_hash_16']
pq_nn
short_config_fname_arr = ['1_hnsw_16', '1_kmeans_independent_16', '1_knn_16',
                          '1_lsh_16', '1_random_hash_16']

counting_index
short_config_fname_arr = ['1_kmeans_independent_16', '1_kmeans_multiple_16', '1_e2lsh_16']
'''

if __name__ == '__main__':
    ds_fname = 'siftsmall'
    _type = 'nn_classification'  # pq_nn nn_classification counting_index
    base_config_dir = '/home/zhengbian/NN_as_Classification/config/%s/small_ds/' % _type
    long_config_dir = base_config_dir + ds_fname + '.json'

    short_config_fname_arr = [10**i for i in range(15)]
    # short_config_fname_arr = ['1_knn_16',
    #                           '8_knn_16']
    for tmp in short_config_fname_arr:
        fname = '8_knn_16_increase_weight_%d.json' % tmp
        short_config_dir = base_config_dir + fname
        # run_nohup(long_config_dir, short_config_dir, ds_fname, tmp, _type)
        run_frontend(long_config_dir, short_config_dir, _type)
    # send_email.send("glove increase weight complete")
