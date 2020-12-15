from procedure_nn_classification.dataset_partition_1.learn_on_graph import learn_on_graph
from procedure_nn_classification.init_0 import multiple_base_partition
import copy


class MultipleLearnOnGraph(multiple_base_partition.MultipleBasePartition):

    def __init__(self, config):
        super(MultipleLearnOnGraph, self).__init__(config)
        self.obj_id = 'learn_on_graph'

    def get_model(self, config):
        return learn_on_graph.LearnOnGraph(config)

    def _preprocess(self, base):
        pass
