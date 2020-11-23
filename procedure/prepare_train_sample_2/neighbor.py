from procedure.prepare_train_sample_2 import base_data_node
from util.vecs import vecs_util
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset


class NeighborDataNode(base_data_node.BaseDataNode):

    def __init__(self, config):
        super(NeighborDataNode, self).__init__(config)
        # self.type, self.save_dir, self.entity_number, self.classifier_number, self.output_type

    '''
    Input:
    -base
    -partition_info
    Output:
    -已经打包好的训练集以及测试集
    '''

    def prepare(self, base, partition_info):
        print('start prepare data %s %s' % (
            self.obj_id, self.output_type))
        # self.datanode
        partition = partition_info[0]
        # 取前label_k个gnd作为训练集的标签
        ranks = vecs_util.get_gnd_numpy(base, base, self.label_k)
        base_idx = torch.arange(0, base.shape[0])
        partition = torch.LongTensor(partition)
        datalen = len(partition)
        cur_split = int(datalen * self.train_split)
        ranks = torch.from_numpy(ranks)

        partition_exp = partition.unsqueeze(0).expand(datalen, -1)
        # datalen x opt.k (or the number of nearest neighbors to take for computing acc)
        neigh_cls = torch.gather(partition_exp, 1, ranks)
        '''
        neigh_cls是一个二维数组
        存放的是每一个节点中相邻节点的partition信息
        每一行的最后一个再加上自己的partition信息
        5000 * 51
        '''
        neigh_cls = torch.cat((neigh_cls, partition.unsqueeze(-1)), dim=1)
        cls_ones = torch.ones(datalen, neigh_cls.size(-1))
        cls_distr = torch.zeros(datalen, self.n_cluster)
        '''
        每一个节点的相邻节点在每一个partition的分布, 包括自己
        每一行相加为51
        '''
        cls_distr.scatter_add_(1, neigh_cls, cls_ones)
        cls_distr /= neigh_cls.size(-1)

        trainset = TensorDataset(base_idx[:cur_split], partition[:cur_split], cls_distr[:cur_split])
        self.trainloader = DataLoader(dataset=trainset, batch_size=self.batch_size,
                                      shuffle=self.shuffle)

        # validation set
        valset = TensorDataset(base_idx[cur_split:], partition[cur_split:])
        self.valloader = DataLoader(dataset=valset, batch_size=self.batch_size, shuffle=False)
        print('finish prepare_data %s %s' % (
            self.obj_id, self.output_type))
        return self.trainloader, self.valloader
