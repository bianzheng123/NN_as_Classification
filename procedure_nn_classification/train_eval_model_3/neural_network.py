from procedure_nn_classification.train_eval_model_3 import classifier
from procedure_nn_classification.train_eval_model_3 import networks
import multiprocessing
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset

lr = 0.008
acc_threshold = 0.95
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class NeuralNetwork(classifier.Classifier):

    def __init__(self, config):
        super(NeuralNetwork, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.n_cluster
        torch.set_num_threads(multiprocessing.cpu_count() // 10 * 9)
        model_config = {
            'n_input': config['n_input'],
            'n_output': self.n_cluster,
            'data_fname': config['data_fname'],
            'distance_metric': config['distance_metric']
        }
        if 'n_character' in config:
            model_config['n_character'] = config['n_character']
        self.model = nn.DataParallel(model_factory(model_config).to(device))
        self.n_epochs = config['n_epochs']
        self.acc_threshold = acc_threshold
        milestones = [10, 17, 24, 31, 38, 45, 50, 55, 60, 70]
        weight_decay = 10 ** (-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.21)
        # the shape of base, which is used in the function eval() to create the score_table
        self.base_shape = None
        self.intermediate_config['train_intermediate'] = []

    def _train(self, base, trainset):
        self.base_shape = base.shape
        base = torch.from_numpy(base)
        trainloader, valloader = trainset
        self.model.train()

        base = base.to(device)

        y = trainloader.dataset.tensors[1]
        # count the number of neighbor point for each label in different bucket
        self.candidate_count_l = [(y == i).sum() for i in range(self.n_cluster)]

        X = []
        for epoch in range(self.n_epochs):
            correct = 0
            loss_l = []
            for i, data_blob in enumerate(trainloader):
                ds_idx, partition, neighbor_distribution = data_blob
                if i == 0:
                    X.extend(list(ds_idx))
                if len(ds_idx) == 1:
                    # since can't batchnorm over a batch of size 1
                    continue
                batch_sz = len(ds_idx)

                ds_idx = ds_idx.to(device)
                partition = partition.to(device)
                neighbor_distribution = neighbor_distribution.to(device)

                ds = base[ds_idx]
                predicted = self.model(ds)

                # count cross entropy
                pred = torch.log(predicted).unsqueeze(-1)
                loss = -torch.bmm(neighbor_distribution.unsqueeze(1), pred).sum()
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
                    ds_idx = ds_idx.to(device)
                    partition = partition.to(device)
                    neighbor_distribution = neighbor_distribution.to(device)
                    cur_data = base[ds_idx]
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
        query = torch.tensor(query)
        query = query.to(device)
        eval_res = None
        with torch.no_grad():
            eval_res = self.model(query)
            self.result = eval_res.cpu().numpy()


def model_factory(config):
    if config['distance_metric'] == 'l2':
        return networks.NNModel(config)
    elif config['distance_metric'] == 'string':
        if config['data_fname'] == 'uniref' or config['data_fname'] == 'unirefsmall':
            return networks.UnirefCNN(config)
    raise Exception("not support the distance metric or dataset")
