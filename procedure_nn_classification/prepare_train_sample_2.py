import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
from util import dir_io

label_k = 50
batch_size = 32
shuffle = True
train_split = 0.99
save_sample = True

def prepare_train(base, base_base_gnd, partition_info, config):
    save_dir = '%s/Classifier_%d/prepare_train_sample' % (
        config['program_train_para_dir'], config['classifier_number'])
    dir_io.mkdir(save_dir)
    classifier_number = config['classifier_number']

    prepare_method = factory(config['type'])
    global label_k
    if 'label_k' in config:
        label_k = config['label_k']
        print(
            "label k %d ==========================================================================================" % label_k)

    start_time = time.time()
    print('start prepare data %s' % classifier_number)
    trainloader, valloader = prepare_method(base, base_base_gnd, partition_info, config['n_cluster'])
    print('finish prepare_data %s' % classifier_number)
    end_time = time.time()
    intermediate_config = {
        'time': end_time - start_time
    }
    if save_sample:
        save(trainloader, valloader, save_dir)

    return (trainloader, valloader), intermediate_config


def save(train_loader, val_loader, save_dir):
    train_save_dir = '%s/trainloader.pth' % save_dir
    dir_io.save_pytorch(train_loader, train_save_dir)

    val_save_dir = '%s/valloader.pth' % save_dir
    dir_io.save_pytorch(val_loader, val_save_dir)
    print("save train sample %s" % save_dir)


def factory(_type):
    if _type == 'neighbor':
        return neighbor
    elif _type == 'neighbor_weight':
        return neighbor_weight
    raise Exception('do not support the type of training data preparation')


def neighbor(base, base_base_gnd, partition_info, n_cluster):
    # datanode
    partition = partition_info[0]
    # extract the top label_k of gnd as the label of training set
    if label_k > base_base_gnd.shape[1]:
        print("\033[32;1m Warning! the shape of base_base_gnd is not enough for label_k \033[0m")
    ranks = base_base_gnd[:, :label_k]

    ranks = ranks.astype(np.int64)
    base_idx = torch.arange(0, base.shape[0])
    partition = torch.LongTensor(partition)
    datalen = len(partition)
    cur_split = int(datalen * train_split)
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
    cls_distr = torch.zeros(datalen, n_cluster)
    '''
    cls_distr means the distribution for neighboring point in each node, including itself
    '''
    cls_distr.scatter_add_(1, neigh_cls, cls_ones)
    cls_distr /= neigh_cls.size(-1)

    trainset = TensorDataset(base_idx[:cur_split], partition[:cur_split], cls_distr[:cur_split])
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=True)

    # validation set
    valset = TensorDataset(base_idx[cur_split:], partition[cur_split:], cls_distr[cur_split:])
    valloader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, drop_last=True)
    return trainloader, valloader


def neighbor_weight(base, base_base_gnd, partition_info, n_cluster):
    # datanode
    partition = partition_info[0]
    # extract the top label_k of gnd as the label of training set
    if label_k > base_base_gnd.shape[1]:
        print("\033[32;1m Warning! the shape of base_base_gnd is not enough for label_k \033[0m")
    ranks = base_base_gnd[:, :label_k]

    weight_range = [10, 20, 30, 40, 50]
    weight_l = [10, 0.4, 0.3, 0.2, 0.1]

    ranks = ranks.astype(np.int64)
    base_idx = torch.arange(0, base.shape[0])
    partition = torch.LongTensor(partition)
    datalen = len(partition)
    cur_split = int(datalen * train_split)
    ranks = torch.from_numpy(ranks)

    partition_exp = partition.unsqueeze(0).expand(datalen, -1)
    neigh_cls = torch.gather(partition_exp, 1, ranks)
    '''
    neigh_cls is a 2d array, stores the partition label of every neighbor node, its order is the order of neighborhood
    '''

    cls_weight = torch.empty(size=[neigh_cls.size(-1)], dtype=torch.float)
    last_idx = 0
    cls_norm = 0
    for i in range(len(weight_l)):
        if weight_range[i] > label_k:
            print("Warning, the weight_range is larger than label_k")
            break
        cls_weight[last_idx:weight_range[i]] = weight_l[i]
        cls_norm += (weight_range[i] - last_idx) * weight_l[i]
        last_idx = weight_range[i]
    cls_weight = cls_weight.expand(datalen, -1)

    cls_distr = torch.zeros(datalen, n_cluster)
    '''
    cls_distr means the distribution for neighboring point in each node, including itself
    '''
    cls_distr.scatter_add_(1, neigh_cls, cls_weight)
    cls_distr /= cls_norm

    trainset = TensorDataset(base_idx[:cur_split], partition[:cur_split], cls_distr[:cur_split])
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size,
                             shuffle=shuffle, drop_last=True)

    # validation set
    valset = TensorDataset(base_idx[cur_split:], partition[cur_split:], cls_distr[cur_split:])
    valloader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, drop_last=True)
    return trainloader, valloader
