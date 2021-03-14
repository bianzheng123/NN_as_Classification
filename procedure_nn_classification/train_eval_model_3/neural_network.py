from procedure_nn_classification.train_eval_model_3 import classifier
from procedure_nn_classification.train_eval_model_3 import networks
import multiprocessing
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
from util import send_email

acc_threshold = 0.95
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.set_num_threads(multiprocessing.cpu_count())

network_config_m = {
    'two_block_512_dim': networks.TwoBlock512Dim,
    'two_block_1024_dim': networks.TwoBlock1024Dim,
    'one_block_2048_dim': networks.OneBlock2048Dim,
    'one_block_8192_dim': networks.OneBlock8192Dim,
    'one_block_512_dim': networks.OneBlock512Dim,
    'two_block_512_dim_no_bn_dropout': networks.TwoBlock512DimNoBnDropout,
    'two_block_8192_dim_no_bn_dropout': networks.TwoBlock8192DimNoBnDropout,
    'res_net': networks.ResNet,
    'cnn': networks.CNN
}


def model_factory(config):
    if config['distance_metric'] == 'l2':
        if config['data_fname'] == 'imagenetsmall' or config['data_fname'] == 'imagenet':
            return 'two_block_8192_dim_no_bn_dropout'
        return 'one_block_8192_dim'
    elif config['distance_metric'] == 'string':
        return 'cnn'
    raise Exception("not support the distance metric or dataset")


def parameter_factory(dataset_partition_method, distance_metric, data_fname, model_name):
    # config['dataset_partition_method'], config['distance_metric'], config['data_fname']
    milestones = [3, 7]
    lr = 0.008
    n_epochs = 12
    if dataset_partition_method == 'knn_random_projection' or dataset_partition_method == 'knn_kmeans' or dataset_partition_method == 'knn_lsh':
        lr = 0.004
        pass
    if data_fname == 'gist':
        lr = 0.00002
    if model_name == 'two_block_512_dim':
        pass
    elif model_name == 'two_block_1024_dim':
        pass
    elif model_name == 'one_block_2048_dim':
        if data_fname == 'sift' and dataset_partition_method == 'knn_kmeans_multiple':
            lr = 0.005
        pass
    elif model_name == 'one_block_8192_dim':
        lr = 0.001
        pass
    elif model_name == 'one_block_512_dim':
        pass
    elif model_name == 'two_block_512_dim_no_bn_dropout':
        lr = 0.0001
        pass
    elif model_name == 'res_net':
        pass
    elif model_name == 'two_block_8192_dim_no_bn_dropout':
        milestones = [5, 7]
        pass
    elif model_name == 'cnn':
        lr = 0.0005
        pass
    else:
        raise Exception('not support model_name')
    return lr, n_epochs, milestones


class NeuralNetwork(classifier.Classifier):

    def __init__(self, config):
        super(NeuralNetwork, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.n_cluster
        # choose by default
        model_name = model_factory(config)
        lr, self.n_epochs, milestones = parameter_factory(config['dataset_partition_method'], config['distance_metric'],
                                                          config['data_fname'], model_name)
        # choose by self config
        if 'model_name' in config:
            model_name = config['model_name']
        if 'lr' in config:
            lr = config['lr']
        if 'n_epochs' in config:
            self.n_epochs = config['n_epochs']
        if 'milestones' in config:
            milestones = config['milestones']

        print('model_name: {}, lr: {}, n_epochs: {}, milestones: {}'.format(model_name, lr,
                                                                            self.n_epochs, milestones))
        self.intermediate_config['train_config'] = {
            'model_name': model_name,
            'lr': lr,
            'n_epochs': self.n_epochs,
            'milestones': milestones
        }

        model_config = {
            'n_input': config['n_input'],
            'n_output': self.n_cluster,
            'data_fname': config['data_fname'],
            'distance_metric': config['distance_metric']
        }
        self.model = nn.DataParallel(network_config_m[model_name](model_config).to(device))

        if 'n_character' in config:
            model_config['n_character'] = config['n_character']

        self.acc_threshold = acc_threshold
        weight_decay = 0
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # the shape of base, which is used in the function eval() to create the score_table
        self.base_shape = None
        self.intermediate_config['train_intermediate'] = []

    def _train(self, base, trainset):
        self.base_shape = base.shape
        base = torch.from_numpy(base)
        trainloader, valloader = trainset

        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        self.model.train()

        # base = base.to(device)

        y = trainloader.dataset.tensors[1]
        # count the number of neighbor point for each label in different bucket
        self.candidate_count_l = [(y == i).sum() for i in range(self.n_cluster)]

        for epoch in range(self.n_epochs):
            correct = 0
            loss_l = []
            for i, data_blob in enumerate(trainloader):
                ds_idx, partition, neighbor_distribution = data_blob
                batch_sz = len(ds_idx)

                # ds_idx = ds_idx.to(device)
                partition = partition.to(device)
                neighbor_distribution = neighbor_distribution.to(device)

                ds = base[ds_idx].float()
                ds = ds.to(device)
                predicted = self.model(ds)

                # count cross entropy
                pred = torch.log(predicted).unsqueeze(-1)
                loss = -torch.bmm(neighbor_distribution.unsqueeze(1), pred).sum()
                if torch.isnan(loss):
                    send_email.send("train parameter contains nan, hostname %s" % send_email.get_host_name())
                    raise Exception("train parameter contains nan")
                loss_l.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                correct += (predicted.argmax(dim=1) == partition).sum().item()
            cur_recall = correct / len(trainloader.dataset)

            # count recall from the extracted dataset in base
            val_cor = 0
            eval_loss_l = []
            self.model.eval()
            with torch.no_grad():
                for i, (ds_idx, partition, neighbor_distribution) in enumerate(valloader):
                    partition = partition.to(device)
                    neighbor_distribution = neighbor_distribution.to(device)
                    cur_data = base[ds_idx].to(device)
                    predicted = self.model(cur_data)
                    pred = torch.log(predicted).unsqueeze(-1)
                    loss = -torch.bmm(neighbor_distribution.unsqueeze(1), pred).sum()
                    eval_loss_l.append(loss.item())
                    val_cor += (predicted.argmax(dim=1) == partition).sum().item()
                val_recall = val_cor / len(valloader.dataset)

            print(
                'epoch {} loss: {} eval_loss: {} train recall: {}    val recall: {} lr: {}'.format(epoch,
                                                                                                   np.mean(loss_l),
                                                                                                   np.mean(eval_loss_l),
                                                                                                   cur_recall,
                                                                                                   val_recall,
                                                                                                   self.optimizer.param_groups[
                                                                                                       0][
                                                                                                       'lr']))
            self.intermediate_config['train_intermediate'].append({
                'epoch': epoch,
                'loss': np.mean(loss_l),
                'eval_loss': np.mean(eval_loss_l),
                'train_recall': cur_recall,
                'val_recall': val_recall,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            self.model.train()
            if cur_recall > self.acc_threshold:
                print('Stopping training as acc is now {}'.format(cur_recall))
                break
            self.scheduler.step()
        print('correct {} Final recall: {}'.format(correct, cur_recall))
        self.intermediate_config['train_total'] = {
            'correct': correct,
            'final_recall': cur_recall
        }

    def _eval(self, query):
        query = torch.tensor(query).float()
        query = query.to(device)
        eval_res = None
        with torch.no_grad():
            eval_res = self.model(query)
            self.result = eval_res.cpu().numpy()
