from procedure.dataset_partition_1.learn_on_graph import learn_on_graph
from procedure.init_0 import multiple_base_partition
import os
import copy


class MultipleLearnOnGraph(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(MultipleLearnOnGraph, self).__init__(config)

    def get_model(self, config):
        return learn_on_graph.LearnOnGraph(config)
