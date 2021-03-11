import minimal_pytorch_rasterizer as mpr
import torch
import cv2
import numpy as np

dtype = torch.float32
device = torch.device('cuda:0')

vertices = torch.tensor([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 1, 1], [1, 1, 1], [1, 0, 1], [0, 0, 1]
    ], dtype=dtype, device=device)

faces = torch.tensor([
        [0, 2, 1], [0, 3, 2], [2, 3, 4], [2, 4, 5], [1, 2, 5], [1, 5, 6],
        [0, 7, 4], [0, 4, 3], [5, 4, 7], [5, 7, 6], [0, 6, 7], [0, 1, 6]
    ], dtype=torch.int32, device=device)

R = torch.tensor(cv2.Rodrigues(np.array([0.5, 0.8, 0.2]))[0], dtype=dtype, device=device)
t = torch.tensor([-0.5, -0.5, 1.3], dtype=dtype, device=device)
vertices = vertices @ R.T + t

pinhole2d = mpr.Pinhole2D(
    fx=250, fy=200,
    cx=160, cy=120,
    w=320, h=240,
)

z_buffer = mpr.project_mesh(
    vertices=vertices,
    faces=faces,
    vertice_values=vertices[:, [2]],  # take z coordinate as values
    pinhole=pinhole2d
)
vis_z_buffer_cpu = mpr.vis_z_buffer(z_buffer)
cv2.imwrite(f'./depth.png', vis_z_buffer_cpu)


coords, normals = mpr.estimate_normals(
    vertices=vertices,
    faces=faces,
    pinhole=pinhole2d
)
vis_normals_cpu = mpr.vis_normals(coords, normals)
cv2.imwrite(f'./normals.png', vis_normals_cpu)
