import run

if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/run/'
    long_dir = 'long_term_config.json'
    short_dir_arr = ['short_term_config.json']
    # 出现bug的配置'siftsmall_8_kmeans_multiple_self_impl.json'

    long_dir = config_dir + long_dir
    for i in range(len(short_dir_arr)):
        short_dir_arr[i] = config_dir + short_dir_arr[i]

    # def delete_dir_if_exist(dir):
    #     if not os.path.isfile(dir):
    #         print(dir)
    #         print("false")
    # for ele in short_dir_arr:
    #     delete_dir_if_exist(ele)
    # delete_dir_if_exist(long_dir)

    for short_dir in short_dir_arr:
        run.run(long_dir, short_dir)
