import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--long_term_config_dir', required=True, help='directory of long_term_config', type=str)
    parser.add_argument('--short_term_config_dir', required=True, help='directory of short_term_config', type=str)
    parser.add_argument('--type', required=True, help='type of different idea', type=str)
    parser.add_argument('--k', required=True, help='at most 50', type=int)

    args = parser.parse_args()
    long_term_config_dir = args.long_term_config_dir
    short_term_config_dir = args.short_term_config_dir
    _type = args.type
    k = args.k
    if _type == 'nn_classification':
        from procedure_nn_classification import run
        run.run(long_term_config_dir, short_term_config_dir, k)
    elif _type == 'pq_nn':
        from procedure_pq_nn import run

        run.run(long_term_config_dir, short_term_config_dir, k)
    elif _type == 'counting_index':
        from procedure_counting_index import run

        run.run(long_term_config_dir, short_term_config_dir, k)
    else:
        raise Exception("not support algorithm type")
