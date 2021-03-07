import torch
from point_cloud_utils.cuda.rasterizer import z_filter as z_filter_cuda
from point_cloud_utils.cuda.rasterizer import estimate_normals as estimate_normals_cuda
from point_cloud_utils.cuda.rasterizer import project_mesh as project_mesh_cuda
from point_cloud_utils.cuda.rasterizer import project_vertices_2d as project_vertices_2d_cuda
from point_cloud_utils.cuda.rasterizer import project_vertices_3d as project_vertices_3d_cuda
from point_cloud_utils import assert_utils


def z_filter(vertices, faces):
    return z_filter_cuda(vertices, faces)


def estimate_normals(vertices, faces, pinhole, vertices_filter=None):
    if vertices_filter is None:
        assert_utils.is_cuda_tensor(vertices)
        assert_utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n), dtype=torch.uint8, device=vertices.device)
    coords, normals = estimate_normals_cuda(
        vertices.contiguous(), faces, vertices_filter,
        pinhole.fx, pinhole.fy,
        pinhole.cx, pinhole.cy,
        pinhole.h, pinhole.w
    )
    return coords, normals


def project_mesh(vertices, faces, vertice_values, pinhole, vertices_filter=None):
    if vertices_filter is None:
        assert_utils.is_cuda_tensor(vertices)
        assert_utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n), dtype=torch.uint8, device=vertices.device)
    return project_mesh_cuda(
        vertices.contiguous(), faces, vertice_values, vertices_filter,
        pinhole.fx, pinhole.fy,
        pinhole.cx, pinhole.cy,
        pinhole.h, pinhole.w
    )


def project_vertices_2d(vertices, vertice_values, pinhole, vertices_filter=None):
    if vertices_filter is None:
        assert_utils.is_cuda_tensor(vertices)
        assert_utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n), dtype=torch.uint8, device=vertices.device)
    return project_vertices_2d_cuda(
        vertices.contiguous(), vertice_values, vertices_filter,
        pinhole.fx, pinhole.fy, 
        pinhole.cx, pinhole.cy,
        pinhole.h, pinhole.w
    )   


def project_vertices_3d(vertices, vertice_values, pinhole_3d, vertices_filter=None):
    if vertices_filter is None:
        assert_utils.is_cuda_tensor(vertices)
        assert_utils.check_shape_len(vertices, 2)
        n = vertices.shape[0]
        vertices_filter = torch.ones((n), dtype=torch.uint8, device=vertices.device)
    return project_vertices_3d_cuda(
        vertices.contiguous(), vertice_values, vertices_filter,
        pinhole_3d.fx, pinhole_3d.fy, pinhole_3d.fz,
        pinhole_3d.cx, pinhole_3d.cy, pinhole_3d.cz,
        pinhole_3d.dimx, pinhole_3d.dimy, pinhole_3d.dimz
    )
