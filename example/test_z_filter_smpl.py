import point_cloud_utils

import torch

from skimage import io
from pose import Pose
import numpy as np
import argparse
import time
import pickle
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
    with open('smpl_tmp.pkl', 'rb') as f:
        mesh_data = pickle.load(f)
    vertices_cpu = mesh_data['vertices']
    faces_cpu = mesh_data['faces']
    uv_cpu = mesh_data['uv']

    pinhole = point_cloud_utils.Pinhole2D(
        fx=500, fy=500, 
        cx=500, cy=500,
        h=1000, w=1000
    )

    pose = Pose(
        R=[
            [1., 0., 0.],
            [0., -1., 0.],
            [0., 0., -1.]
        ],
        t=[
            [0.],
            [-0.3],
            [1.2]
        ]
    )

    t_hist = defaultdict(list)
    t_names = [
        'cpu->gpu_transfer',
        'apply_transform',
        'z_filter',
        'project_vertices_2d',
        'gpu->cpu_transfer',
    ]
    for it in range(repeat):
        t0 = time.time()
        vertices = torch.tensor(vertices_cpu, dtype=torch.float32, device='cuda:0')
        vertice_values = torch.tensor(uv_cpu, dtype=torch.float32, device='cuda:0')
        faces = torch.tensor(faces_cpu, dtype=torch.int32, device='cuda:0')
        
        t1 = time.time()
        vertices_pose = point_cloud_utils.apply_transform(vertices, pose.R, pose.t)

        t2 = time.time()
        vertices_filter = point_cloud_utils.z_filter(vertices_pose, faces)

        t3 = time.time()
        projected = point_cloud_utils.project_vertices_2d(vertices_pose, vertice_values, pinhole, vertices_filter)

        t4 = time.time()
        img = (projected * 255).cpu().numpy().round().clip(0, 255).astype(np.uint8)

        t5 = time.time()

        save_t(t_hist, t_names, [t0, t1, t2, t3, t4, t5])

        if it == 0:
            io.imsave('./tmp_u.png', img[:, :, 0])
            io.imsave('./tmp_v.png', img[:, :, 1])
    
    if repeat > 1:
        print_t(t_hist, t_names)


def main():
    parsed_args = parse_args()
    test_z_filter(repeat=parsed_args.repeat)


if __name__ == '__main__':
    main()
