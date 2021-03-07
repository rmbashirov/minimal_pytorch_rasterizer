import numpy as np
from point_cloud_utils.cuda.transformer import apply_transform as apply_transform_cuda


def apply_transform(vertices, R, t, z_mean=0, alpha=1, scale_z_only=False):
    R = np.array(R, dtype=np.float32).flatten().tolist()
    t = np.array(t, dtype=np.float32).flatten().tolist()
    return apply_transform_cuda(vertices, R, t, z_mean, alpha, scale_z_only)
