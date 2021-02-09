import _init_paths
import os


def run_nohup(method, dataset, n_codebook):
    os.system(
        'nohup python3 -u baseline/%s.py --dataset %s --k_gnd 10 --metric euclid_norm --num_codebook %d '
        '--num_cluster 256 > ./log/baseline/%s_%s.log 2>&1 &' % (
            method, dataset, n_codebook, dataset, n_codebook))


if __name__ == '__main__':
    dataset_l = ['deep', 'gist', 'glove', 'imagenet', 'sift']
    k_gnd = 10
    codebook_l = [1, 2, 4, 8]
    n_cluster = 256
    metric = 'euclid_norm'

    for ds in dataset_l:
        for n_codebook in codebook_l:
            run_nohup('run_pq', ds, n_codebook)
            run_nohup('run_opq', ds, n_codebook)
