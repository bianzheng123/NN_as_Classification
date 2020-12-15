import json
import os


def save_json(save_dir, result_fname, json_file):
    with open('%s/%s' % (save_dir, result_fname), 'w') as f:
        json.dump(json_file, f)


def save_config(config):
    save_dir = '%s/config' % config['save_dir']
    os.system('mkdir %s' % save_dir)

    long_config = config['long_term_config']
    short_config = config['short_term_config']
    short_config_before_run = config['short_term_config_before_run']
    intermediate_result = config['intermediate_result']

    save_json(save_dir, 'long_term_config.json', long_config)
    save_json(save_dir, 'short_term_config.json', short_config)
    save_json(save_dir, 'short_term_config_before_run.json', short_config_before_run)
    save_json(config['save_dir'], 'intermediate_result.json', intermediate_result)
    print('save program: %s' % config['program_fname'])
