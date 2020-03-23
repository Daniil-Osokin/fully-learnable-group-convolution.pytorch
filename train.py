import argparse
import os

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models.mobilenet_v2 import MobileNetV2
from test import test


def train(checkpoint_path, num_groups, use_standard_group_convolutions, checkpoints_folder, num_epochs_to_dump_net):
    net = MobileNetV2(groups_in_1x1=num_groups, use_flgc=(not use_standard_group_convolutions))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250, 350])
    best_accuracy = 0
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        best_accuracy = checkpoint['best_accuracy']
        start_epoch = checkpoint['last_epoch'] + 1

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    for epoch_id in range(start_epoch, 400):
        print('Epoch: {}'.format(epoch_id))
        if num_epochs_to_dump_net is not None:
            if epoch_id % num_epochs_to_dump_net == 0:
                torch.save({'net': net}, os.path.join(checkpoints_folder, 'net_epoch_{}.pth'.format(epoch_id)))
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()
            if (batch_idx+1) % 100 == 0:
                print('Batch: {}, loss: {}'.format(batch_idx + 1, loss.item()))

        print('Testing...')
        accuracy = test(net, testloader)
        print('Accuracy: {}%'.format(accuracy))
        lr_scheduler.step()

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({'state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'best_accuracy': best_accuracy,
                        'last_epoch': epoch_id},
                       os.path.join(checkpoints_folder, 'best_checkpoint.pth'))
        torch.save({'state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'best_accuracy': best_accuracy,
                    'last_epoch': epoch_id},
                   os.path.join(checkpoints_folder, 'last_checkpoint.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Trains MobileNet V2 with different types of group convolutions\n'
                                     'in place of 1x1 convolutions on CIFAR-10 dataset. It compares\n'
                                     'fully learnable group convolution and standard group convolution.')
    parser.add_argument('--checkpoint-path', type=str, default=None, help='checkpoint path to continue training from')
    parser.add_argument('--num-groups', type=int, default=8, help='group number in group convolutions')
    parser.add_argument('--use-standard-group-convolutions', action='store_true', default=False,
                        help='use standard group convolutions instead of fully learnable')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='experiment name to create folder for checkpoints')
    parser.add_argument('--num-epochs-to-dump-net', type=int, default=None, help='number of epochs to dump network to futher analyze it')
    args = parser.parse_args()

    checkpoints_folder = '{}_checkpoints'.format(args.experiment_name)
    if not os.path.exists(checkpoints_folder):
        os.makedirs(checkpoints_folder)

    train(args.checkpoint_path, args.num_groups, args.use_standard_group_convolutions, checkpoints_folder, args.num_epochs_to_dump_net)
