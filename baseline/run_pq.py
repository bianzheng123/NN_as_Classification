import _init_paths
from sorter import *
from util.vecs import vecs_io
from util import dir_io


def chunk_compress(pq, vecs):
    chunk_size = 1000000
    compressed_vecs = np.empty(shape=vecs.shape, dtype=np.float32)
    for i in tqdm.tqdm(range(math.ceil(len(vecs) / chunk_size))):
        compressed_vecs[i * chunk_size: (i + 1) * chunk_size, :] \
            = pq.compress(vecs[i * chunk_size: (i + 1) * chunk_size, :].astype(dtype=np.float32))
    return compressed_vecs


def execute(pq, X, Q, G, metric, config, train_size=100000):
    np.random.seed(123)
    print("# ranking metric {}".format(metric))
    print("# " + pq.class_message())
    pq.fit(X[:train_size].astype(dtype=np.float32), iter=40)

    print('# compress items')
    compressed = chunk_compress(pq, X)
    print(compressed.dtype)
    print("# sorting items")
    Ts = [2 ** i for i in range(1 + int(math.log2(len(X))))]
    recalls = BatchSorter(compressed, Q, X, G, Ts, metric=metric, batch_size=200).recall()
    print("# searching!")

    res_l = []
    # print("expected items, overall time, avg recall, avg precision, avg error, avg items")
    for i, (t, recall) in enumerate(zip(Ts, recalls)):
        tmp_res = {
            'n_candidate': t,
            'recall': recall
        }
        res_l.append(tmp_res)
        # print("{}, {}, {}, {}, {}, {}".format(
        #     2 ** i, 0, recall, recall * len(G[0]) / t, 0, t))
    save_data_dir = '/home/zhengbian/NN_as_Classification/data/result/%s_%d_baseline_%d_pq' % (
        config['dataset'], config['n_cluster'], config['codebook'])
    dir_io.delete_dir_if_exist(save_data_dir)
    dir_io.mkdir(save_data_dir)
    dir_io.save_json(save_data_dir, 'result.json', res_l)


def parse_args():
    # override default parameters with command line parameters
    import argparse
    parser = argparse.ArgumentParser(description='Process input method and parameters.')
    parser.add_argument('--dataset', type=str, help='choose data set name')
    parser.add_argument('--k_gnd', type=int, help='required topk of ground truth')
    parser.add_argument('--metric', type=str, help='metric of ground truth')
    parser.add_argument('--num_codebook', type=int, help='number of codebooks')
    parser.add_argument('--num_cluster', type=int, help='number of centroids in each quantizer')
    args = parser.parse_args()
    return args.dataset, args.k_gnd, args.num_codebook, args.num_cluster, args.metric


# python3 run_pq --dataset siftsmall --k_gnd 10 --metric euclid_norm --num_codebook 1 --num_cluster 16
# python3 run_pq --dataset siftsmall --k_gnd 10 --metric euclid_norm --num_codebook 8 --num_cluster 16

if __name__ == '__main__':
    dataset = 'glove'
    k_gnd = 10
    codebook = 4
    n_cluster = 256
    metric = 'euclid_norm'

    # override default parameters with command line parameters
    import sys

    if len(sys.argv) > 3:
        dataset, k_gnd, codebook, n_cluster, metric = parse_args()
    else:
        import warnings

        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, k_gnd = {}, codebook = {}, n_cluster = {}, metric = {}"
          .format(dataset, k_gnd, codebook, n_cluster, metric))

    load_data_config = {
        'data_dir': 'data/dataset/%s_%d' % (dataset, k_gnd)
    }
    X, Q, G, base_base_gnd = vecs_io.read_data_l2(load_data_config)
    print(len(X))
    # pq, rq, or component of norm-pq
    quantizer = PQ(M=codebook, Ks=n_cluster)
    exe_config = {
        'dataset': dataset,
        'codebook': codebook,
        'n_cluster': n_cluster
    }
    execute(quantizer, X, Q, G, metric, exe_config)
