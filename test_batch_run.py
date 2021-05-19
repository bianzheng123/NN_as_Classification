import os
from util import send_email


def run_nohup(long_config_dir, short_config_dir, dataset_name, fname, _type, k):
    os.system('nohup python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type %s --k %d > '
              './log/%s/%s_%d_%s.log 2>&1 &' % (
                  long_config_dir, short_config_dir, _type, k, _type, dataset_name, k, fname))


def run_frontend(long_config_dir, short_config_dir, _type, k):
    os.system('python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type %s --k %d' % (
        long_config_dir, short_config_dir, _type, k))


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
    # deep gist glove imagenet sift
    ds_fname_l = ['siftsmall']
    k_l = [5]
    _type = 'counting_index'  # pq_nn nn_classification counting_index
    base_config_dir = '/home/zhengbian/NN_as_Classification/config/%s/small_ds/' % _type
    for ds_fname in ds_fname_l:
        for k in k_l:
            long_config_dir = base_config_dir + ds_fname + '.json'

            short_config_fname_arr = ['4_kmeans_multiple_16']

            for tmp_fname in short_config_fname_arr:
                short_config_dir = base_config_dir + tmp_fname + '.json'
                # run_nohup(long_config_dir, short_config_dir, ds_fname, tmp_fname, _type, k)
                run_frontend(long_config_dir, short_config_dir, _type, k)

    # send_str = "complete the work in " + " ".join(ds_fname_l) + " " + send_email.get_host_name()
    # send_email.send(send_str)
