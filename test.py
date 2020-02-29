import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from models.mobilenet_v2 import MobileNetV2


def test(net, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    accuracy = correct / total * 100
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint-path', type=str, default=None, help='checkpoint path to continue training from')
    parser.add_argument('--num-groups', type=int, default=8, help='group number in group convolutions')
    parser.add_argument('--use-standard-group-convolutions', action='store_true', default=False,
                        help='use standard group convolutions instead of fully learnable')
    args = parser.parse_args()

    net = MobileNetV2(groups_in_1x1=args.num_groups, use_flgc=(not args.use_standard_group_convolutions))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    net.load_state_dict(checkpoint['state_dict'])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2)

    print('Testing...')
    accuracy = test(net, dataloader)
    print('Accuracy: {}%'.format(accuracy))
