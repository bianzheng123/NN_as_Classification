import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--long_term_config_dir', required=True, help='directory of long_term_config', type=str)
    parser.add_argument('--short_term_config_dir', required=True, help='directory of short_term_config', type=str)
    parser.add_argument('--type', required=True, help='type of different idea', type=str)

    args = parser.parse_args()
    long_term_config_dir = args.long_term_config_dir
    short_term_config_dir = args.short_term_config_dir
    _type = args.type
    if _type == 'nn_classification':
        from procedure_nn_classification import run
        run.run(long_term_config_dir, short_term_config_dir)
    elif _type == 'pq_nn':
        from procedure_pq_nn import run

        run.run(long_term_config_dir, short_term_config_dir)
    elif _type == 'counting_index':
        from procedure_counting_index import run

        run.run(long_term_config_dir, short_term_config_dir)
    else:
        raise Exception("not support algorithm type")
