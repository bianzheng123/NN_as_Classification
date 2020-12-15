from procedure.prepare_train_sample_2 import base_data_node
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from util.vecs import vecs_util


class NeighborDataNode(base_data_node.BaseDataNode):

    def __init__(self, config):
        super(NeighborDataNode, self).__init__(config)
        # self.type, self.save_dir, self.entity_number, self.classifier_number, self.output_type

    def _prepare(self, base, base_base_gnd, partition_info):
        # self.datanode
        partition = partition_info[0]
        # extract the top label_k of gnd as the label of training set
        ranks = base_base_gnd[:, :self.label_k]
        # ranks = vecs_util.get_gnd_numpy(base, base, self.label_k)

        base_idx = torch.arange(0, base.shape[0])
        partition = torch.LongTensor(partition)
        datalen = len(partition)
        cur_split = int(datalen * self.train_split)
        ranks = torch.from_numpy(ranks)

        partition_exp = partition.unsqueeze(0).expand(datalen, -1)
        # datalen x opt.k (or the number of nearest neighbors to take for computing acc)
        neigh_cls = torch.gather(partition_exp, 1, ranks)
        '''
        neigh_cls is a 2d array, stores the partition label of every neighbor node for a single node
        for each row, the last position is added by the self partition label
        '''
        neigh_cls = torch.cat((neigh_cls, partition.unsqueeze(-1)), dim=1)
        cls_ones = torch.ones(datalen, neigh_cls.size(-1))
        cls_distr = torch.zeros(datalen, self.n_cluster)
        '''
        cls_distr means the distribution for neighboring point in each node, including itself
        '''
        cls_distr.scatter_add_(1, neigh_cls, cls_ones)
        cls_distr /= neigh_cls.size(-1)

        trainset = TensorDataset(base_idx[:cur_split], partition[:cur_split], cls_distr[:cur_split])
        self.trainloader = DataLoader(dataset=trainset, batch_size=self.batch_size,
                                      shuffle=self.shuffle)

        # validation set
        valset = TensorDataset(base_idx[cur_split:], partition[cur_split:], cls_distr[cur_split:])
        self.valloader = DataLoader(dataset=valset, batch_size=self.batch_size, shuffle=False)
