import _init_paths
import traceback
from procedure_nn_classification.train_eval_model_3 import neural_network
from procedure_nn_classification.train_eval_model_3 import networks
from util.vecs import vecs_io
import torch
import numpy as np
import torch.nn as nn
import multiprocessing
from util import send_email

start_lr = 0.006
lr_gap = 0.001

n_epochs = 12
milestones = [3, 7]

distance_metric = 'l2'  # string
acc_threshold = 0.95
model_name = 'one_block_2048_dim'
base_dir = '/home/zhengbian/NN_as_Classification/data/dataset/siftsmall_10/base.fvecs'
# base_dir = '/home/zhengbian/NN_as_Classification/data/dataset/sift_10/base.fvecs'
dataset, dimension = vecs_io.fvecs_read(base_dir)

# train_sample_dir = '/home/zhengbian/NN_as_Classification/data/train_para/sift_256_nn_1_kmeans_multiple_/Classifier_0' \
#                    '/prepare_train_sample'
train_sample_dir = '/home/zhengbian/NN_as_Classification/data/train_para/siftsmall_16_nn_1_knn_preconfiguration_fastsocial/Classifier_0' \
                   '/prepare_train_sample'

model_config = {
    'n_input': dimension,  # dimension of base
    'n_output': 16,  # n_cluster
    'data_fname': 'knn',
    'distance_metric': 'l2'
}

network_m = {
    'two_block_512_dim': networks.TwoBlock512Dim,
    'two_block_1024_dim': networks.TwoBlock1024Dim,
    'one_block_2048_dim': networks.OneBlock2048Dim,
    'one_block_512_dim': networks.OneBlock512Dim,
    'two_block_512_dim_no_bn_dropout': networks.TwoBlock512DimNoBnDropout,
    'two_block_8192_dim_no_bn_dropout': networks.TwoBlock8192DimNoBnDropout,
    'res_net': networks.ResNet,
    'cnn': networks.CNN
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(100)
torch.cuda.manual_seed_all(100)
torch.set_num_threads(multiprocessing.cpu_count())


def prune_lr(learning_rate, trainset, base):
    trainloader, valloader = trainset
    model = nn.DataParallel(network_m[model_name](model_config).to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    base = torch.from_numpy(base)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

    model.train()

    # base = base.to(device)

    y = trainloader.dataset.tensors[1]
    final_loss = []
    for epoch in range(n_epochs):
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
            predicted = model(ds)

            # count cross entropy
            pred = torch.log(predicted).unsqueeze(-1)
            loss = -torch.bmm(neighbor_distribution.unsqueeze(1), pred).sum()
            if torch.isnan(loss):
                raise Exception("train parameter contains nan")
            loss_l.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (predicted.argmax(dim=1) == partition).sum().item()
        cur_recall = correct / len(trainloader.dataset)

        # count recall from the extracted dataset in base
        val_cor = 0
        eval_loss_l = []
        model.eval()
        with torch.no_grad():
            for i, (ds_idx, partition, neighbor_distribution) in enumerate(valloader):
                partition = partition.to(device)
                neighbor_distribution = neighbor_distribution.to(device)
                cur_data = base[ds_idx].to(device)
                predicted = model(cur_data)
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
                                                                                               optimizer.param_groups[
                                                                                                   0][
                                                                                                   'lr']))
        if epoch == n_epochs - 1:
            final_loss = [np.mean(loss_l), np.mean(eval_loss_l)]
        model.train()
        if cur_recall > acc_threshold:
            print('Stopping training as acc is now {}'.format(cur_recall))
            break
        scheduler.step()
    return final_loss


if __name__ == '__main__':
    trainloader_dir = '%s/trainloader.pth' % train_sample_dir
    trainloader = torch.load(trainloader_dir)
    valloader_dir = '%s/valloader.pth' % train_sample_dir
    valloader = torch.load(valloader_dir)

    current_lr = start_lr - lr_gap
    greatest_lr = -1
    min_loss = [1000000, 1000000] # first is train loss, second is eval loss
    try:
        while True:
            print("now learning rate: %.5f" % (current_lr + lr_gap))
            tmp_loss = prune_lr(current_lr + lr_gap, (trainloader, valloader), dataset)
            if tmp_loss[1] < min_loss[1]:
                min_loss = tmp_loss
                greatest_lr = current_lr + lr_gap
            current_lr += lr_gap
            print("tmp best loss: train %.2f eval %.2f, greatest_lr %.5f\n" % (min_loss[0], min_loss[1], greatest_lr))
    except Exception as e:
        traceback.print_exc()
        print("greatest lr %.5f" % greatest_lr)
        print("biggest lr %.5f" % current_lr)
        print("best lr loss: train %.2f eval %.2f" % (min_loss[0], min_loss[1]))
