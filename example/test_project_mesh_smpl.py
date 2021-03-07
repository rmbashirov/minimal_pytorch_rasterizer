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
        print(f'\t{t_name}: '.ljust(max_t_names_len + 1) + f'{mean_str} +- {std_str}ms')
    # print(f'skip_counts: {skip_counts}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--double', action='store_true')
    parsed_args = parser.parse_args()
    return parsed_args


def calculate_timings_0(vertices_cpu, uv_cpu, faces_cpu, pinhole, repeat, dtype=torch.float32):
    t_hist = defaultdict(list)
    t_names = [
        'cpu->gpu_transfer',
        'project_mesh',
        'gpu->cpu_transfer',
    ]

    for it in range(repeat):
        t0 = time.time()
        vertices = torch.tensor(vertices_cpu, dtype=dtype, device='cuda:0')
        vertice_values = torch.tensor(uv_cpu, dtype=dtype, device='cuda:0')
        faces = torch.tensor(faces_cpu, dtype=torch.int32, device='cuda:0')

        t1 = time.time()
        projected = point_cloud_utils.project_mesh(vertices, faces, vertice_values, pinhole)

        t2 = time.time()
        if it == 0:
            img = (projected * 255).cpu().numpy().round().clip(0, 255).astype(np.uint8)
        else:
            img = projected[:5, :5, :].cpu().numpy()

        t3 = time.time()

        if it > 0:
            save_t(t_hist, t_names, [t0, t1, t2, t3])

        if it == 0:
            io.imsave('./tmp_u.png', img[:, :, 0])
            io.imsave('./tmp_v.png', img[:, :, 1])

    if repeat > 1:
        print('method 1 timings:')
        print_t(t_hist, t_names)


def calculate_timings_1(vertices_cpu, uv_cpu, faces_cpu, pinhole, repeat, dtype=torch.float32):
    for it in range(repeat + 1):
        if it == 0:
            vertices = torch.tensor(vertices_cpu, dtype=dtype, device='cuda:0')
            vertice_values = torch.tensor(uv_cpu, dtype=dtype, device='cuda:0')
            faces = torch.tensor(faces_cpu, dtype=torch.int32, device='cuda:0')
        elif it == 1:
            start = time.time()

        projected = point_cloud_utils.project_mesh(vertices, faces, vertice_values, pinhole)
        torch.cuda.synchronize()

        if it == 0:
            img = (projected * 255).cpu().numpy().round().clip(0, 255).astype(np.uint8)

            io.imsave('./tmp_u.png', img[:, :, 0])
            io.imsave('./tmp_v.png', img[:, :, 1])

    torch.cuda.synchronize()
    end = time.time()
    timing = (end - start) / (repeat - 1) * 1000
    print(f'method 2 timings:\n\t{timing:.3f}ms')


def test_project_mesh(repeat, double=False):
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

    vertices_cpu = pose.R @ np.swapaxes(vertices_cpu, 0, 1) + pose.t
    vertices_cpu = np.swapaxes(vertices_cpu, 0, 1)

    dtype = torch.float64 if double else torch.float32
    calculate_timings_0(vertices_cpu, uv_cpu, faces_cpu, pinhole, repeat, dtype=dtype)
    if repeat > 1:
        calculate_timings_1(vertices_cpu, uv_cpu, faces_cpu, pinhole, repeat, dtype=dtype)


def main():
    parsed_args = parse_args()
    test_project_mesh(repeat=parsed_args.repeat, double=parsed_args.double)


if __name__ == '__main__':
    main()
