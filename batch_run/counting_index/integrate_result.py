import os


def run_nohup(config_dir, fname):
    os.system(
        'nohup python3 integrate_result_counting_index.py --integrate_result_config_dir %s > ./log/%s.log 2>&1 &' % (
            config_dir, fname))


def run_frontend(config_dir):
    os.system('python3 integrate_result_counting_index.py --integrate_result_config_dir %s' % config_dir)


if __name__ == '__main__':
    config_sub_dir = '/home/bz/NN_as_Classification/config/counting_index/integrate_result_3/specific_config/siftsmall/'
    config_fname_l = ['siftsmall_1_kmeans', 'siftsmall_1_kmeans_multiple', 'siftsmall_1_lsh',
                      'siftsmall_8_kmeans', 'siftsmall_8_kmeans_multiple', 'siftsmall_8_lsh']
    for fname in config_fname_l:
        dire = config_sub_dir + fname + '.json'
        # run_nohup(dire, fname + '_integrate_result')
        run_frontend(dire)
