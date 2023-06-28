from __future__ import print_function
import os
import time
from PIL import Image
from loss_plm import peer_learning_loss
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils import *


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((550, 550)),
        transforms.RandomCrop(448, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root='./bird/train', transform=transform_train)
    print(len(trainset))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Model
    if resume:
        net = torch.load(model_path)
    else:
        net = PMG()
    netp  = net
    # netp = torch.nn.DataParallel(net, device_ids=[0,1])

    # GPU
    device = torch.device("cuda:0")
    net.to(device)
    # cudnn.benchmark = True

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD([{'params': net.parameters(), 'lr': 0.002}],
        momentum=0.9, weight_decay=5e-4)

    max_val_acc = 0
    lr = [0.002]
    for epoch in range(start_epoch, nb_epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        train_loss1 = 0
        train_loss2 = 0
        train_loss3 = 0
        train_loss4 = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            iter_start_time = time.time()
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rate
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

            # # Step 1
            # optimizer.zero_grad()
            # inputs1 = jigsaw_generator(inputs, 8)
            # output_1, _, _, _ = netp(inputs1)
            # loss1 = CELoss(output_1, targets) * 1
            # loss1.backward()
            # optimizer.step()
            #
            # # Step 2
            # optimizer.zero_grad()
            # inputs2 = jigsaw_generator(inputs, 4)
            # _, output_2, _, _ = netp(inputs2)
            # loss2 = CELoss(output_2, targets) * 1
            # loss2.backward()
            # optimizer.step()
            #
            # # Step 3
            # optimizer.zero_grad()
            # inputs3 = jigsaw_generator(inputs, 2)
            # _, _, output_3, _ = netp(inputs3)
            # loss3 = CELoss(output_3, targets) * 1
            # loss3.backward()
            # optimizer.step()

            # Step 4
            optimizer.zero_grad()
            logits_1, logits_2, xlayer1, xlayer2, concat = netp(inputs)
            loss1 = peer_learning_loss(logits_1, logits_2,targets, xlayer1, xlayer2)
            loss1.backward()
            optimizer.step()

            # optimizer.zero_grad()
            # output1, output_image1, concat = netp(inputs)
            # loss2 = peer_learning_loss(output1, targets, output_image1)
            # loss2.backward()
            # optimizer.step()

            #  training log
            _, predicted = torch.max(concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # train_loss += (loss1.item() + loss2.item())
            train_loss += loss1.item()
            # train_loss1 += loss1.item()
            # train_loss2 += loss2.item()
            iter_end_time = time.time()
            a = iter_end_time - iter_start_time
            if batch_idx % 50 == 0:
                print(
                    'Step: %d | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f | Loss: %.3f | Acc: %.3f%% (%d/%d)|time: %.5f' % (
                    batch_idx, train_loss1 / (batch_idx + 1), train_loss2 / (batch_idx + 1),
                    train_loss3 / (batch_idx + 1), train_loss4 / (batch_idx + 1), train_loss / (batch_idx + 1),
                    100. * float(correct) / total, correct, total,a))

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f | Loss1: %.3f | Loss2: %.5f | Loss3: %.5f | Loss_concat: %.5f |\n' % (
                epoch, train_acc, train_loss, train_loss1 / (idx + 1), train_loss2 / (idx + 1), train_loss3 / (idx + 1),
                train_loss4 / (idx + 1)))

        if epoch < 5 or epoch >= 20:
            val_acc, val_acc_com, val_loss = test(net, peer_learning_loss, 3)
            if val_acc_com > max_val_acc:
                max_val_acc = val_acc_com
                net.cpu()
                torch.save(net, './' + store_name + '/model.pth')
                net.to(device)
            with open(exp_dir + '/results_test.txt', 'a') as file:
                file.write('Iteration %d, test_acc = %.5f, test_acc_combined = %.5f, test_loss = %.6f\n' % (
                epoch, val_acc, val_acc_com, val_loss))
        else:
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)


train(nb_epoch=200,             # number of epoch
         batch_size=16,         # batch size
         store_name='bird',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path='')         # the saved model where you want to resume the training
