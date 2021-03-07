#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define BLOCK_SIZE_2D_X 32
#define BLOCK_SIZE_2D_Y 16

// vertices coords:
// vertices[:, 0]: x
// vertices[:, 1]: y
// vertices[:, 2]: z

// 2d tensor axis:
// 0: yi
// 1: xi

// kernels

template <typename scalar_t>
__global__ void get_vertices_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> depth,
    const torch::PackedTensorAccessor32<uint8_t,2> mask,
    torch::PackedTensorAccessor32<scalar_t,2> vertices,
    torch::PackedTensorAccessor32<int32_t,2> vertice_indxs,
    scalar_t near, scalar_t far,
    scalar_t fx, scalar_t fy, 
    scalar_t cx, scalar_t cy,
    int * g_v_c
) {
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;

    if (yi >= depth.size(0) || xi >= depth.size(1)) {
        return;
    }
    scalar_t z = depth[yi][xi];
    bool cnd = near < z && z < far;
    if (mask[yi][xi] == 0) {
        cnd = false;
    }
    __shared__ int v_c, v_s;
    int v_i, v_c_i = -1;
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        v_c = 0;
    }
    __syncthreads();

    if (cnd) {
        v_c_i = atomicAdd(&v_c, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        v_s = atomicAdd(g_v_c, v_c);
    }
    __syncthreads();
 
    // set global mem
    if (v_c_i >= 0) {
        v_i = v_s + v_c_i;
        vertices[v_i][0] = z * ((scalar_t)xi - cx) / fx;
        vertices[v_i][1] = z * ((scalar_t)yi - cy) / fy;
        vertices[v_i][2] = z;
        vertice_indxs[yi][xi] = v_i;
    }
}

template <typename scalar_t>
__global__ void get_vertice_values_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,3> values,
    const torch::PackedTensorAccessor32<int32_t,2> vertice_indxs,
    torch::PackedTensorAccessor32<scalar_t,2> vertice_values,
    const int C
) {
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;
    if (xi >= vertice_indxs.size(1) || yi >= vertice_indxs.size(0)) {
        return;
    }

    int vertice_indx = vertice_indxs[yi][xi];
    if (vertice_indx >= 0) {
        for (int c = 0; c < C; c++) {
            vertice_values[vertice_indx][c] = values[yi][xi][c];
        }
    }
}

__global__ void get_faces_cuda_kernel(
    const torch::PackedTensorAccessor32<int32_t,2> vertice_indxs,
    torch::PackedTensorAccessor32<int32_t,2> faces,
    int * g_f_c
) {
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;
    if (xi >= vertice_indxs.size(1) || yi >= vertice_indxs.size(0)) {
        return;
    }
    __shared__ int f_c, f_s;
    int f_i;
    int f_c_i_00, f_c_i_01, f_c_i_10, f_c_i_11;
    f_c_i_00 = f_c_i_01 = f_c_i_10 = f_c_i_11 = -1;
    int i00, i01, i10, i11;
    __shared__ int shared_vertice_indxs[BLOCK_SIZE_2D_Y + 1][BLOCK_SIZE_2D_X + 1];

    // read from global mem to shared mem
    shared_vertice_indxs[threadIdx.y + 1][threadIdx.x + 1] = vertice_indxs[yi][xi];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        f_c = 0;
        if (yi > 0 && xi > 0) {
            shared_vertice_indxs[0][0] = vertice_indxs[yi - 1][xi - 1];
        }
    }
    if (threadIdx.x == 0 && yi > 0) {
        shared_vertice_indxs[threadIdx.y + 1][0] = vertice_indxs[yi - 1][xi];
    }
    if (threadIdx.y == 0 && xi > 0) {
        shared_vertice_indxs[0][threadIdx.x + 1] = vertice_indxs[yi][xi - 1];
    }
    __syncthreads();

    if (yi > 0 && xi > 0) {
        i00 = shared_vertice_indxs[threadIdx.y][threadIdx.x];
        i01 = shared_vertice_indxs[threadIdx.y][threadIdx.x + 1];
        i10 = shared_vertice_indxs[threadIdx.y + 1][threadIdx.x];
        i11 = shared_vertice_indxs[threadIdx.y + 1][threadIdx.x + 1];

        if (i11 >= 0) {
            if (i00 >= 0) {
                if (i01 >= 0) {
                    f_c_i_01 = atomicAdd(&f_c, 1);
                }
                if (i10 >= 0) {
                    f_c_i_10 = atomicAdd(&f_c, 1);
                }
            } else if (i01 >=0 && i10 >= 0) {
                f_c_i_11 = atomicAdd(&f_c, 1);
            }
        } else {
            if (i00 >= 0 && i01 >= 0 && i10 >= 0) {
                f_c_i_00 = atomicAdd(&f_c, 1);
            }
        }
    }
    __syncthreads();

    if (f_c == 0) {
        return;
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        f_s = atomicAdd(g_f_c, f_c);
    }
    __syncthreads();

    // set global mem
    // add triangles in clock-wise order
    if (f_c_i_01 >= 0) {
        f_i = f_s + f_c_i_01;
        faces[f_i][0] = i11;
        faces[f_i][1] = i01;
        faces[f_i][2] = i00;
    }
    if (f_c_i_10 >= 0) {
        f_i = f_s + f_c_i_10;
        faces[f_i][0] = i11;
        faces[f_i][1] = i00;
        faces[f_i][2] = i10;
    }
    if (f_c_i_11 >= 0) {
        f_i = f_s + f_c_i_11;
        faces[f_i][0] = i11;
        faces[f_i][1] = i01;
        faces[f_i][2] = i10;
    }
    if (f_c_i_00 >= 0) {
        f_i = f_s + f_c_i_00;
        faces[f_i][0] = i00;
        faces[f_i][1] = i10;
        faces[f_i][2] = i01;
    }
}

// intermediate functions

std::tuple<torch::Tensor, torch::Tensor> get_vertices_cuda(
    const torch::Tensor& depth, 
    const torch::Tensor& mask, 
    float near, float far,
    float fx, float fy, 
    float cx, float cy
) {
    const int H = depth.size(0);
    const int W = depth.size(1);

    const dim3 dimBlock(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y);
    const dim3 dimGrid((W - 1) / dimBlock.x + 1, (H - 1) / dimBlock.y + 1);

    int * g_v_c;
    cudaMalloc((void**)&g_v_c, sizeof(int));
    cudaMemset(g_v_c, 0, sizeof(int));

    const int gpuid = depth.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto vertices = torch::zeros(
        {H * W, 3}, 
        torch::dtype(depth.scalar_type()).device(torch::kCUDA, gpuid)
    );
    auto vertice_indxs = torch::ones(
        {H, W}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA, gpuid)
    ) * -1;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(depth.scalar_type(), "get_vertices_cuda_kernel", [&] {
        get_vertices_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            depth.packed_accessor32<scalar_t,2>(),
            mask.packed_accessor32<uint8_t,2>(),
            vertices.packed_accessor32<scalar_t,2>(),
            vertice_indxs.packed_accessor32<int32_t,2>(),
            (scalar_t)near, (scalar_t)far,
            (scalar_t)fx, (scalar_t)fy, (scalar_t)cx, (scalar_t)cy,
            g_v_c
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    int N;
    cudaMemcpy(&N, g_v_c, sizeof(int), cudaMemcpyDeviceToHost);
    vertices.resize_({N, 3});

    return std::tuple<torch::Tensor, torch::Tensor>{vertices, vertice_indxs};
}

torch::Tensor get_vertice_values_cuda(const torch::Tensor& values, const torch::Tensor& vertice_indxs, int N){
    const int H = vertice_indxs.size(0);
    const int W = vertice_indxs.size(1);
    const int C = values.size(2);

    const dim3 dimBlock(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y);
    const dim3 dimGrid((W - 1) / dimBlock.x + 1, (H - 1) / dimBlock.y + 1);

    const int gpuid = values.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto vertice_values = torch::zeros(
        {N, C}, 
        torch::dtype(values.scalar_type()).device(torch::kCUDA, gpuid)
    );
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.scalar_type(), "get_vertice_values_cuda_kernel", [&] {
        get_vertice_values_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
            values.packed_accessor32<scalar_t,3>(),
            vertice_indxs.packed_accessor32<int32_t,2>(),
            vertice_values.packed_accessor32<scalar_t,2>(),
            C
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return vertice_values;
}

torch::Tensor get_faces_cuda(const torch::Tensor& vertice_indxs) {
    const int H = vertice_indxs.size(0);
    const int W = vertice_indxs.size(1);

    const dim3 dimBlock(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y);
    const dim3 dimGrid((W - 1) / dimBlock.x + 1, (H - 1) / dimBlock.y + 1);

    int * g_f_c;
    cudaMalloc((void**)&g_f_c, sizeof(int));
    cudaMemset(g_f_c, 0, sizeof(int));

    const int gpuid = vertice_indxs.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto faces = torch::zeros(
        {H * W * 4, 3}, 
        torch::dtype(torch::kInt32).device(torch::kCUDA, gpuid)
    );

    get_faces_cuda_kernel<<<dimGrid, dimBlock>>>(
        vertice_indxs.packed_accessor32<int32_t,2>(),
        faces.packed_accessor32<int32_t,2>(),
        g_f_c
    );
    AT_CUDA_CHECK(cudaGetLastError());

    int M;
    cudaMemcpy(&M, g_f_c, sizeof(int), cudaMemcpyDeviceToHost);
    faces.resize_({M, 3});

    return faces;
}

// cpp defined functions

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_mesh_cuda(
    const torch::Tensor& depth, 
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float near, float far,
    float fx, float fy, 
    float cx, float cy
) {
    const int gpuid = depth.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto get_vertices_cuda_res = get_vertices_cuda(
        depth, mask,
        near, far,
        fx, fy, 
        cx, cy
    );
    auto vertices = std::get<0>(get_vertices_cuda_res);
    auto vertice_indxs = std::get<1>(get_vertices_cuda_res);

    auto vertice_values = get_vertice_values_cuda(values, vertice_indxs, vertices.size(0));

    auto faces = get_faces_cuda(vertice_indxs);

    return std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>{vertices, vertice_values, faces};
}

std::tuple<torch::Tensor, torch::Tensor> create_point_cloud_cuda(
    const torch::Tensor& depth, 
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float near, float far,
    float fx, float fy, 
    float cx, float cy
) {
    const int gpuid = depth.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto get_vertices_cuda_res = get_vertices_cuda(
        depth, mask,
        near, far,
        fx, fy, 
        cx, cy
    );
    auto vertices = std::get<0>(get_vertices_cuda_res);
    auto vertice_indxs = std::get<1>(get_vertices_cuda_res);

    auto vertice_values = get_vertice_values_cuda(values, vertice_indxs, vertices.size(0));

    return std::tuple<torch::Tensor, torch::Tensor>{vertices, vertice_values};
}
