'''
input base and output the text information of partition
'''


def partition(base, model, base_base_gnd, obj):
    partition_info, model_info, para = model.partition(base, base_base_gnd, obj)
    # model.save()
    return partition_info, model_info, para
