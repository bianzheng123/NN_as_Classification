import os


def run_nohup(long_config_dir, short_config_dir, dataset_fname, fname):
    os.system(
        'nohup python3 -u run.py --long_term_config_dir %s --short_term_config_dir %s --type counting_index > '
        './log/counting_index/%s_%s.log 2>&1 &' % (
            long_config_dir, short_config_dir, dataset_fname, fname))


def run_frontend(long_config_dir, short_config_dir):
    os.system('python3 run.py --long_term_config_dir %s --short_term_config_dir %s --type counting_index' % (
        long_config_dir, short_config_dir))


if __name__ == '__main__':
    ds_fname = 'siftsmall'
    base_config_dir = '/home/bianzheng/NN_as_Classification/config/counting_index/%s/' % ds_fname
    long_config_dir = base_config_dir + 'long_term_config.json'

    short_config_fname_arr = ['1_kmeans_16', '1_kmeans_multiple_16', '1_lsh_16',
                              '8_kmeans_16', '8_kmeans_multiple_16', '8_lsh_16']
    # short_config_fname_arr = ['1_kmeans_16']
    for tmp_fname in short_config_fname_arr:
        short_config_dir = base_config_dir + tmp_fname + '.json'
        run_nohup(long_config_dir, short_config_dir, ds_fname, tmp_fname)
        # run_frontend(long_config_dir, short_config_dir)