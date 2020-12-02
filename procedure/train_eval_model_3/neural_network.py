from procedure.train_eval_model_3 import classifier
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset


class NeuralNetwork(classifier.Classifier):

    def __init__(self, config):
        super(NeuralNetwork, self).__init__(config)
        # self.type, self.save_dir, self.classifier_number, self.n_cluster
        config['network']['n_output'] = self.n_cluster
        self.model = NNModel(config['network'])
        self.n_epochs = config['n_epochs']
        self.acc_threshold = config['acc_threshold']
        lr = config['lr']
        milestones = [10, 17, 24, 31, 38, 45, 50, 55, 60, 70]
        weight_decay = 10 ** (-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.21)
        # 用于记录base的形状, eval时生成score table会用到
        self.base_shape = None
        self.intermediate_config['train_intermediate'] = []

    def _train(self, base, trainset):
        self.base_shape = base.shape
        base = torch.from_numpy(base)
        trainloader, valloader = trainset
        self.model.train()

        y = trainloader.dataset.tensors[1]
        # 计数每一个标签不同桶中点的数量
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

                ds = base[ds_idx]
                predicted = self.model(ds)

                # 计算交叉熵
                pred = torch.log(predicted).unsqueeze(-1)
                loss = -torch.bmm(neighbor_distribution.unsqueeze(1), pred).sum()
                loss_l.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                correct += (predicted.argmax(dim=1) == partition).sum().item()
            cur_recall = correct / len(trainloader.dataset)

            # 从抽出的数据集计算recall
            val_cor = 0
            self.model.eval()
            with torch.no_grad():
                for i, (ds_idx, partition) in enumerate(valloader):
                    cur_data = base[ds_idx]
                    predicted = self.model(cur_data)
                    val_cor += (predicted.argmax(dim=1) == partition).sum().item()
                val_recall = val_cor / len(valloader.dataset)

            # print(
            #     'epoch {} loss: {} train recall: {}    val recall: {} lr: {}'.format(epoch, np.mean(loss_l), cur_recall,
            #                                                                          val_recall,
            #                                                                          self.optimizer.param_groups[0][
            #                                                                              'lr']))
            self.intermediate_config['train_intermediate'].append({
                'epoch': epoch,
                'loss': np.mean(loss_l),
                'train_recall': cur_recall,
                'val_recall': val_recall,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            self.model.train()
            if cur_recall > self.acc_threshold:
                print('Stopping training as acc is now {}'.format(cur_recall))
                break
            self.scheduler.step()
        # print('correct {} Final recall: {}'.format(correct, cur_recall))
        self.intermediate_config['train_total'] = {
            'correct': correct,
            'final_recall': cur_recall
        }

    def _eval(self, query, label_map):
        query = torch.tensor(query)
        eval_res = None
        with torch.no_grad():
            eval_res = self.model(query)

        score_table = np.zeros((query.numpy().shape[0], self.base_shape[0]))
        # 对每一个query加分
        for j in range(query.shape[0]):
            # 对每一个cluster加分
            for k in range(eval_res.size(1)):
                score_item_idx_l = label_map[k]
                score_table[j][score_item_idx_l] = eval_res[j][k].item()
        self.result = score_table


class NNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_dim = config['n_input']
        hidden_dim_1 = config['n_hidden'][0]
        hidden_dim_2 = config['n_hidden'][1]
        output_dim = config['n_output']
        dropout_probability = config['p_dropout']
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.BatchNorm1d(hidden_dim_1),
            nn.ReLU(),

            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.BatchNorm1d(hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),

            nn.Linear(hidden_dim_2, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.layer(x)
        return x
