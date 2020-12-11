'''
input base and output the text information of partition
'''


def partition(base, model):
    partition_info, model_info = model.partition(base)
    # model.save()
    return partition_info, model_info
