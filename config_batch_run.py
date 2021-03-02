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
    ds_fname = 'sift'
    _type = 'nn_classification'  # pq_nn nn_classification counting_index
    base_config_dir = '/home/zhengbian/NN_as_Classification/config/%s/big_ds/' % _type
    long_config_dir = base_config_dir + ds_fname + '.json'

    # para_l = ['two_block_512_dim', 'res_net', 'one_block_2048_dim'] # cnn 'two_block_8192_dim_no_bn_dropout'
    # para_l = ['two_block_512_dim', 'two_block_1024_dim', 'one_block_2048_dim', 'one_block_512_dim',
    #           'two_block_512_dim_no_bn_dropout', 'res_net']  # cnn two_block_8192_dim_no_bn_dropout
    para_l = ['two_block_512_dim_no_bn_dropout']  # cnn two_block_8192_dim_no_bn_dropout
    # para_l = [1, 2, 3]
    method_l = ['knn_random_projection']
    para_name = 'model'
    n_classifier = 4
    for method in method_l:
        for para in para_l:
            fname = '{}_{}_256_{}_{}.json'.format(n_classifier, method, para_name, para)
            short_config_dir = base_config_dir + fname
            # run_nohup(long_config_dir, short_config_dir, ds_fname, fname, _type)
            run_frontend(long_config_dir, short_config_dir, _type)
    # send_email.send("glove increase weight complete")
