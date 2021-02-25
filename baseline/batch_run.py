import _init_paths
import os


def run_nohup(method, dataset, n_codebook, n_cluster):
    log_fname = '%s_%d_%d' % (dataset, n_codebook, n_cluster)
    os.system(
        'nohup python3 -u baseline/%s.py --dataset %s --k_gnd 10 --metric euclid_norm --num_codebook %d '
        '--num_cluster %d > ./log/baseline/%s.log 2>&1 &' % (
            method, dataset, n_codebook, n_cluster, log_fname))


if __name__ == '__main__':
    dataset_l = ['deep', 'gist', 'glove', 'imagenet', 'sift']
    k_gnd = 10
    codebook_l = [8]
    n_cluster = 16
    metric = 'euclid_norm'

    for ds in dataset_l:
        for n_codebook in codebook_l:
            run_nohup('run_pq', ds, n_codebook, n_cluster)
            # run_nohup('run_opq', ds, n_codebook)
