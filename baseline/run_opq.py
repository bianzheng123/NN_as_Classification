from run_pq import execute
from opq import OPQ
from run_pq import parse_args
import _init_paths
from util.vecs import vecs_io
from util import dir_io


if __name__ == '__main__':
    dataset = 'normalsmall'
    k_gnd = 10
    codebook = 2
    n_cluster = 16
    metric = 'euclid_norm'

    # override default parameters with command line parameters
    import sys
    if len(sys.argv) > 3:
        dataset, k_gnd, codebook, n_cluster, metric = parse_args()
    else:
        import warnings
        warnings.warn("Using  Default Parameters ")
    print("# Parameters: dataset = {}, topK = {}, codebook = {}, n_cluster = {}, metric = {}"
          .format(dataset, k_gnd, codebook, n_cluster, metric))

    load_data_config = {
        'data_dir': 'data/dataset/%s_%d' % (dataset, k_gnd)
    }
    X, Q, G, base_base_gnd = vecs_io.read_data_l2(load_data_config)
    # pq, rq, or component of norm-pq
    quantizer = OPQ(M=codebook, Ks=n_cluster)
    exe_config = {
        'dataset': dataset,
        'codebook': codebook,
        'n_cluster': n_cluster,
        'method': 'opq'
    }
    execute(quantizer, X, Q, G, metric, exe_config)