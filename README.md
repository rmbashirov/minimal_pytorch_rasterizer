## About
**minimal_pytorch_rasterizer** is a CUDA non-differentiable mesh rasterization library for pytorch tensors with python bindings.

It projects mesh to image using pinhole camera model. Vertices could have any number of features (channels). Library also estimates normals for mesh visualization.

A mesh with 6890 vertices and 13776 faces is rasterized on 1000x1000 image in less than 1ms on 2080ti GPU. Check timings [here](./example/timing.py).

The results are consistent with [nvdiffrast](https://github.com/NVlabs/nvdiffrast) output. Check comparison [here](./example/nvdiffrast_compare.ipynb). 
  
## Example

[Visualize z buffer and normals of cube](./example/cube.py):
```python
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
cv2.imwrite('./depth.png', vis_z_buffer_cpu)


coords, normals = mpr.estimate_normals(
    vertices=vertices,
    faces=faces,
    pinhole=pinhole2d
)
vis_normals_cpu = mpr.vis_normals(coords, normals)
cv2.imwrite('./normals.png', vis_normals_cpu)
``` 
Will produce:

![result](./example/depth.png)
![result](./example/normals.png)

## Installation
- `pip intall .` or `./setup.sh`
- To build for custom cuda arches set env variable: `export TORCH_CUDA_ARCH_LIST="Pascal Turing"`. This env variable is used [here](https://github.com/pytorch/pytorch/blob/5710374e4e335c6761d2b8b937a2b54a5577cb5d/torch/utils/cpp_extension.py#L1298).
- Possible intallation errors:
    - `packedpacked_accessor32` in error msgs means you have pytorch version < 1.3
    - Errors caused by pytorch internal header files could mean that you have pytorch cuda version (provided by cudatoolkit) and nvcc cuda version mismatch
- Docker environment to run comparison [notebook](./example/nvdiffrast_compare.ipynb) is provided:
  - Install [Docker](https://www.docker.com/), [nvidia-docker](https://github.com/NVIDIA/nvidia-docker), set `nvidia` your default runtime for `docker`
  - Build docker image: `make build`
  - Enter docker container: `make run`
  - Run jupyter
- Tested till pytorch 1.8
- `torch.float32` or `torch.float64` dtypes are supported, `torch.float16` is not
