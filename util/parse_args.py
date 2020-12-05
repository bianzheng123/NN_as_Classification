import argparse

# 建立解析对象
parser = argparse.ArgumentParser()

parser.add_argument('--long_term_config_dir', required=True, help='directory of long_term_config', type=str)
parser.add_argument('--short_term_config_dir', required=True, help='directory of short_term_config', type=str)


# 给xx实例增加一个aa属性

def parse_args():
    # 把parser中设置的所有"add_argument"给返回到args子类实例当中
    # 那么parser中增加的属性内容都会在args实例中，使用即可。
    args = parser.parse_args()
    return args
