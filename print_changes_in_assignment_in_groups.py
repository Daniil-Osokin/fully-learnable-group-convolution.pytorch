import argparse
import glob
import os

import numpy as np
import torch

from modules.flgc import Flgc2d


def get_epoch(checkpoint_path):
    return int(checkpoint_path.split('net_epoch_')[1].split('.pth')[0])


def get_flgc_layer(net, needed_layer_id):
    layer_id = -1
    for module in net.modules():
        if isinstance(module, Flgc2d):
            layer_id += 1
            if layer_id == needed_layer_id:
                return module
    raise IndexError('No flgc layer with such id, possible range: [{}, {}]'.format(0, layer_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Allows to track changes in assignment of input channels/filters in groups')
    parser.add_argument('--checkpoints-dir', type=str, required=True, help='path to the directory with model checkpoints')
    parser.add_argument('--layer-id', type=int, default=0, help='flgc layer id to print statistic')
    parser.add_argument('--track-filters', action='store_true', default=False, help='track changes in assignment of filters in groups')
    args = parser.parse_args()

    checkpoint_paths = glob.glob(os.path.join(args.checkpoints_dir, 'net_epoch_*.pth'))
    checkpoint_paths = sorted(checkpoint_paths, key=lambda path: get_epoch(path))

    net = torch.load(checkpoint_paths[0], map_location='cpu')['net']
    module = get_flgc_layer(net, args.layer_id)
    rows_names = ['{:^13}'.format('epoch {}'.format(get_epoch(checkpoint_paths[0])))]
    assignment_map_to_check = module.out_channels_in_group_assignment_map if args.track_filters else module.in_channels_in_group_assignment_map
    previous_assignment_map = module.binarize(torch.softmax(assignment_map_to_check, dim=1)).detach().data.cpu().numpy()
    channels_per_group = np.sum(previous_assignment_map, axis=0).astype(np.int32)
    cols_names = ['{:^13}'.format('group {}'.format(group_id)) for group_id in range(len(channels_per_group))]
    cols_names = ['{:^13}'.format('')] + cols_names + ['{:^13}'.format('Total diff.')]
    stats = [[] for _ in range(len(channels_per_group) + 1)]
    for group_id, channels_num in enumerate(channels_per_group):
        stats[group_id].append('{:^13}'.format('{:+d}/{:+d}/{:d}'.format(0, 0, channels_num)))
    stats[-1].append('{:^13}'.format(0))
    
    for checkpoint_path in checkpoint_paths[1:]:
        rows_names.append('{:^13}'.format('epoch {}'.format(get_epoch(checkpoint_path))))
        net = torch.load(checkpoint_path, map_location='cpu')['net']
        module = get_flgc_layer(net, args.layer_id)
        assignment_map_to_check = module.out_channels_in_group_assignment_map if args.track_filters else module.in_channels_in_group_assignment_map
        assignment_map = module.binarize(torch.softmax(assignment_map_to_check, dim=1)).detach().data.cpu().numpy()
        assignment_map_diff = assignment_map - previous_assignment_map
        deleted_channels_per_group = assignment_map_diff.copy()
        deleted_channels_per_group[deleted_channels_per_group > 0] = 0
        deleted_channels_per_group = np.sum(deleted_channels_per_group, axis=0).astype(np.int32)
        added_channels_per_group = assignment_map_diff.copy()
        added_channels_per_group[added_channels_per_group < 0] = 0
        added_channels_per_group = np.sum(added_channels_per_group, axis=0).astype(np.int32)
        channels_per_group = np.sum(assignment_map, axis=0).astype(np.int32)
        for group_id, channels_num in enumerate(channels_per_group):
            stats[group_id].append('{:^13}'.format('{:+d}/{:+d}/{:d}'.format(deleted_channels_per_group[group_id],
                                                                             added_channels_per_group[group_id], channels_num)))
        stats[-1].append('{:^13}'.format(np.sum(np.abs(assignment_map_diff).astype(np.int32))))
        previous_assignment_map = assignment_map
    for col_name in cols_names:
        print(col_name, end='')
    print('\n')
    for row_id, row_name in enumerate(rows_names):
        print(row_name, end='')
        for group_stat in stats:
            print(group_stat[row_id], end='')
        print('\n')
