from procedure import run


if __name__ == '__main__':
    config_dir = '/home/bz/NN_as_Classification/config/run_2/'
    long_config_dir = config_dir + 'long_term_config.json'

    short_config_dir = config_dir + 'short_term_config.json'
    run.train_eval(long_config_dir, short_config_dir)
