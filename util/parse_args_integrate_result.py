import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--integrate_result_config_dir', required=True, help='directory of long_term_config', type=str)


def parse_args():
    args = parser.parse_args()
    return args
