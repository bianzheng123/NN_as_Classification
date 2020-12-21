'''
input base and output the text information of partition
'''


def partition(base, model, obj):
    partition_info, model_info, para = model.partition(base, obj)
    # model.save()
    return partition_info, model_info, para
