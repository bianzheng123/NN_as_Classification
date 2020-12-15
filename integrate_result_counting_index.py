from procedure_counting_index import integrate_result
from util import parse_args_integrate_result

if __name__ == '__main__':
    args = parse_args_integrate_result.parse_args()
    integrate_result_config_dir = args.integrate_result_config_dir
    integrate_result.integrate_result(integrate_result_config_dir)
