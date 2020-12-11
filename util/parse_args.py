import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--long_term_config_dir', required=True, help='directory of long_term_config', type=str)
parser.add_argument('--short_term_config_dir', required=True, help='directory of short_term_config', type=str)


def parse_args():
    args = parser.parse_args()
    return args
