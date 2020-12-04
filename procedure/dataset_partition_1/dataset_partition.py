'''
输入base, 输出partition的文本信息
'''


def partition(base, model):
    partition_info, model_info = model.partition(base)
    # model.save()
    return partition_info, model_info
