""" The script is developed to showcase distribution of ML training across
several GPU nodes with Open Data Hub and Kubeflow on OpenShift.

    Parts of the script involving datasets and model training
were adopted from the following blog (and further modified):
 https://yangkky.github.io/2019/07/08/distributed-pytorch-tutorial.html

    Part of the script involving Kubeflow components such as MASTER_ADDR etc.
was developed with the help of and adopted from scripts of:
  Juana Nakfour and Sanjay Arora at Red Hat, Inc.

    The script is designed specifically for GPU nodes. For CPU nodes, modifications
will need to be made. """

import os
from datetime import datetime
import argparse
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist

__author__ = "Selbi Nuryyeva at Red Hat, Inc."

key_list = ['MASTER_ADDR', 'MASTER_PORT', 'RANK', 'WORLD_SIZE']

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


def train(rank, world_size, n_epochs):

    env_dict = {key: os.environ[key] for key in key_list}

    # Required line for DistributedDataParallel (DDP) below
    # It initializes distribution process
    # nccl is recommended for GPUs
    dist.init_process_group(backend='nccl')

    model = ConvNet()
    gpu = torch.cuda.current_device()
    model = model.cuda()

    # Batch size can be modified. 75 was chosen so each GPU gets 100 batches (8 GPUs total)
    # There are a total of 60,000 images in MNIST dataset
    batch_size = 75

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)

    # Wrap the model using DDP: parallelizes training by splitting data across devices
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu], output_device=gpu)

    # Data loading code
    train_dataset = torchvision.datasets.MNIST(root='./',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(n_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format((epoch) + 1, n_epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    n_epochs = 20
    train(rank, world_size, n_epochs)
