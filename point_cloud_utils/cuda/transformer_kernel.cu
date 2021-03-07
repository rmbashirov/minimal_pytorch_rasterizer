#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE 512

// kernels

template <typename scalar_t>
__global__ void apply_transform_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    torch::PackedTensorAccessor32<scalar_t,2> vertices_transformed,
    scalar_t* R, scalar_t* t,
    scalar_t z_mean, scalar_t alpha, bool scale_z_only
) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertices.size(0)) {
        return;
    }
    scalar_t x, y, z, rx, ry, rz, scale_z;
    
    x = vertices[i][0];
    y = vertices[i][1];
    z = vertices[i][2];
    
    rx = x * R[0] + y * R[1] + z * R[2];
    ry = x * R[3] + y * R[4] + z * R[5];
    rz = x * R[6] + y * R[7] + z * R[8];

    if (z_mean > 1e-6) {
        scale_z = rz / z_mean;
        scale_z = (1 - alpha) * 1 + alpha * scale_z;
        if (scale_z_only) {
            rx += t[0];
            ry += t[1];
            rz += scale_z * t[2];
        } else {
            rx += scale_z * t[0];
            ry += scale_z * t[1];
            rz += scale_z * t[2];
        }
    } else {
        rx += t[0];
        ry += t[1];
        rz += t[2];
    }

    vertices_transformed[i][0] = rx;
    vertices_transformed[i][1] = ry;
    vertices_transformed[i][2] = rz;
}

// utils

template <typename T>
struct static_cast_func {
  template <typename T1>
  T operator()(const T1& x) const { return static_cast<T>(x); }
};

// cpp defined functions

torch::Tensor apply_transform_cuda(
    const torch::Tensor& vertices,
    float R[9], float t[3],
    float z_mean, float alpha, bool scale_z_only
) {
    const int gpuid = vertices.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    const int N = vertices.size(0);
    const dim3 dimBlock(BLOCK_SIZE);
    const dim3 dimGrid((N - 1) / dimBlock.x + 1);
    auto vertices_transformed = torch::zeros_like(vertices);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "apply_transform_cuda_kernel", [&] {
        scalar_t R_cast[9];
        std::transform(R, R + 9, R_cast, static_cast_func<scalar_t>());
        scalar_t* R_cast_cuda;
        cudaMalloc(&R_cast_cuda, 9 * sizeof(scalar_t)); 
        cudaMemcpy(R_cast_cuda, R_cast, 9 * sizeof(scalar_t), cudaMemcpyHostToDevice);
        
        scalar_t t_cast[3];
        std::transform(t, t + 3, t_cast, static_cast_func<scalar_t>());
        scalar_t* t_cast_cuda;
        cudaMalloc(&t_cast_cuda, 3 * sizeof(scalar_t)); 
        cudaMemcpy(t_cast_cuda, t_cast, 3 * sizeof(scalar_t), cudaMemcpyHostToDevice);

        apply_transform_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            vertices_transformed.packed_accessor32<scalar_t,2>(),
            R_cast_cuda, t_cast_cuda,
            (scalar_t)z_mean, (scalar_t)alpha, scale_z_only
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return vertices_transformed;
}
