import torch
from point_cloud_utils.cuda.loader import create_mesh as create_mesh_cuda
from point_cloud_utils.cuda.loader import create_point_cloud as create_point_cloud_cuda
from point_cloud_utils import assert_utils


def create_mesh(depth_cuda, values_cuda, near, far, pinhole, mask_cuda=None):
    if mask_cuda is None:
        assert_utils.is_cuda_tensor(depth_cuda)
        assert_utils.check_shape_len(depth_cuda, 2)
        h, w = depth_cuda.shape[:2]
        mask_cuda = torch.ones((h, w), dtype=torch.uint8, device=depth_cuda.device)
    vertices, vertice_values, faces = create_mesh_cuda(
        depth_cuda, values_cuda, mask_cuda,
        near, far,
        pinhole.fx, pinhole.fy, 
        pinhole.cx, pinhole.cy
    )
    return vertices, vertice_values, faces


def create_point_cloud(depth_cuda, values_cuda, near, far, pinhole, mask_cuda=None):
    if mask_cuda is None:
        assert_utils.is_cuda_tensor(depth_cuda)
        assert_utils.check_shape_len(depth_cuda, 2)
        h, w = depth_cuda.shape[:2]
        mask_cuda = torch.ones((h, w), dtype=torch.uint8, device=depth_cuda.device)
    vertices, vertice_values = create_point_cloud_cuda(
        depth_cuda, values_cuda, mask_cuda,
        near, far,
        pinhole.fx, pinhole.fy, 
        pinhole.cx, pinhole.cy
    )
        
    return vertices, vertice_values
