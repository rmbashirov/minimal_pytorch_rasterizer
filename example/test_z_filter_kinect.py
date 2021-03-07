import point_cloud_utils

import torch

from skimage import io
from pose import Pose
import numpy as np
import argparse
import time
from collections import defaultdict


def save_t(t_hist, t_names, ts):
    assert len(ts) - 1 == len(t_names)
    ts = np.array(ts)
    ts = ts[1:] - ts[:-1]
    for i, t_name in enumerate(t_names):
        t_hist[t_name].append(ts[i])


def print_t(t_hist, t_names, skip_fraq=0.1):
    skip_counts = []
    max_t_names_len = max(map(len, t_names)) + 5
    for t_name in t_names:
        dts = t_hist[t_name]
        skip_count = int(len(dts) * skip_fraq)
        skip_counts.append(skip_count)
        dts = dts[skip_count:]
        mean_str = f'{np.mean(dts) * 1000:.2f}'.rjust(6)
        std_str = f'{np.std(dts) * 1000:.2f}'.ljust(6)
        print(f'{t_name}: '.ljust(max_t_names_len + 1) + f'{mean_str} +- {std_str}ms')
    print(f'skip_counts: {skip_counts}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1)
    parsed_args = parser.parse_args()
    return parsed_args


def test_z_filter(repeat):
    azure_color = io.imread('./data/azure/color/00000.jpg')
    azure_depth = io.imread('./data/azure/depth/00000.png').astype(np.float32) / 1000

    azure_pinhole = point_cloud_utils.Pinhole2D(
        fx=899.4442138671875, fy=898.8028564453125, 
        cx=957.82110595703125, cy=548.6478271484375
    )
    azure_near, azure_far = 0.1, 1.5

    train_pinhole = point_cloud_utils.Pinhole2D(
        fx=758.13416, fy=759.1479,
        cx=268.41144, cy=482.12332,
        h=960, w=540
    )

    train_pose = Pose(
        R=[
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]
        ],
        t=[
            [-0.06340354],
            [-0.41022787],
            [-0.05393599]
        ]
    )

    novel_pose = Pose(
        R=[
            [0.76434811, 0.09425269, -0.63787804],
            [-0.4299411, 0.81174853, -0.39524042],
            [0.48054408, 0.57635125, 0.66098156]
        ],
        t=[
            [0.50570373],
            [0.03329333],
            [0.03079345]
        ]
    )
    
    t_hist = defaultdict(list)
    t_names = [
        'cpu->gpu_transfer',
        'create_mesh',
        'apply_transform_novel',
        'apply_transform_train',
        'z_filter',
        'project_vertices_2d',
        'gpu->cpu_transfer',
    ]
    for it in range(repeat):
        t0 = time.time()
        azure_color_cuda = torch.tensor(azure_color, dtype=torch.float32, device='cuda:0')
        azure_depth_cuda = torch.tensor(azure_depth, dtype=torch.float32, device='cuda:0')
        
        t1 = time.time()
        vertices, vertice_values, faces = point_cloud_utils.create_mesh(
            azure_depth_cuda, azure_color_cuda, 
            azure_near, azure_far, 
            azure_pinhole
        )

        t2 = time.time()
        vertices_novel = point_cloud_utils.apply_transform(vertices, novel_pose.R, novel_pose.t)

        t3 = time.time()
        vertices_train = point_cloud_utils.apply_transform(
            vertices, 
            train_pose.R, train_pose.t,
            z_mean=1,
            scale_z_only=False
        )

        t4 = time.time()
        vertices_filter = point_cloud_utils.z_filter(vertices_novel, faces)

        t5 = time.time()
        projected = point_cloud_utils.project_vertices_2d(vertices_train, vertice_values, train_pinhole, vertices_filter)

        t6 = time.time()
        img = projected.cpu().numpy().round().clip(0, 255).astype(np.uint8)

        t7 = time.time()

        save_t(t_hist, t_names, [t0, t1, t2, t3, t4, t5, t6, t7])

        if it == 0:
            io.imsave('./tmp.png', img)
    
    if repeat > 1:
        print_t(t_hist, t_names)


def main():
    parsed_args = parse_args()
    test_z_filter(repeat=parsed_args.repeat)


if __name__ == '__main__':
    main()
