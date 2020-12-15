from procedure_counting_index import run
import os
from util import parse_args

if __name__ == '__main__':
    args = parse_args.parse_args()
    long_term_config_dir = args.long_term_config_dir
    short_term_config_dir = args.short_term_config_dir
    run.run(long_term_config_dir, short_term_config_dir)
