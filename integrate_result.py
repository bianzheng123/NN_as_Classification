from procedure import integrate_result


if __name__ == '__main__':
    config_sub_dir = '/home/bz/NN_as_Classification/config/integrate_result_3/'
    config_fname = 'config.json'
    integrate_result.integrate_result(config_sub_dir + config_fname)
