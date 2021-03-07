#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>

#define BLOCK_SIZE 512
#define BLOCK_SIZE_2D_X 32
#define BLOCK_SIZE_2D_Y 16
#define BLOCK_SIZE_3D_X 32
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 4

// vertices coords:
// vertices[:, 0]: x
// vertices[:, 1]: y
// vertices[:, 2]: z

// 2d tensor axis:
// 0: yi
// 1: xi

// 3d tensor axis:
// 0: zi
// 1: yi
// 2: xi

template <typename scalar_t>
__device__ __forceinline__ scalar_t atomicMinFloat(scalar_t * addr, scalar_t value) {
        scalar_t old;
        old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
             __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));
        return old;
}

__device__ double atomicMin_double(double* address, double val)
{
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// kernel utils

template <typename scalar_t>
__device__ int lower_bound(const scalar_t* values, const scalar_t value, const int N) {
    int left = 0;
    int right = N;
    int mid;
    while (right - left > 1) {
        mid = (left + right) / 2;
        if (values[mid] < value) {
            left = mid;
        } else {
            right = mid;
        }
    }
    return right;
}

// kernels

template <typename scalar_t>
__global__ void z_filter_cuda_kernel_0(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    scalar_t* xs, scalar_t* ys, scalar_t* zs,
    int* indexes
) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertices.size(0)) {
        return;
    }

    const scalar_t eps = 1e-6;
    scalar_t z = vertices[i][2];
    if (abs(z) < eps) {
        z = -1;
    }
    const scalar_t z_inv = 1. / z;
    xs[i] = vertices[i][0] * z_inv;
    ys[i] = vertices[i][1] * z_inv;
    zs[i] = z;
    indexes[i] = i;
}

template <typename scalar_t>
__global__ void z_filter_cuda_kernel_1(
    const scalar_t* xs, const scalar_t* ys, const scalar_t* zs,
    const scalar_t* xs_sort, const int* indexes,
    const torch::PackedTensorAccessor32<int32_t,2> faces,
    torch::PackedTensorAccessor32<uint8_t,1> vertices_filter
) {
    // blockIdx, threadIdx
    // gridDim, blockDim

    // const int face_indx = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    const int face_indx = blockIdx.x;

    if (face_indx >= faces.size(0)) {
        return;
    }

    __shared__ int left, right, vertices_per_thread;
    __shared__ int ai, bi, ci;
    __shared__ bool is_bad_face;
    __shared__ double col_sum_x, col_sum_y, col_sum_z_inv;
    __shared__ scalar_t min_x, max_x, min_y, max_y;
    __shared__ double m[9];
    __shared__ double m_adj[9];
    __shared__ double m_adj_scalar_t[9];
    __shared__ double det, det_inv;
    __shared__ double col_sum_z;
    const double eps_det = 1e-9;
    const scalar_t eps = 1e-6;
    const scalar_t eps_border = 1e-1; // NEGATIVE value considers points on edges to be OUTSIDE triangels
    const scalar_t eps_border_left_0 = -eps_border;
    const scalar_t eps_border_right_1 = 1 + eps_border;

    if (threadIdx.x == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        m[0] = xs[ai]; m[1] = xs[bi]; m[2] = xs[ci];
        m[3] = ys[ai]; m[4] = ys[bi]; m[5] = ys[ci];
        m[6] = zs[ai]; m[7] = zs[bi]; m[8] = zs[ci];

        if (m[6] < eps or m[7] < eps or m[8] < eps) {
            is_bad_face = true;
            return;
        }
        is_bad_face = false;
    }
    __syncthreads();
    if (is_bad_face) {
        return;
    }

    if (threadIdx.x == 0) {
        m_adj[0] = m[4] * m[8] - m[7] * m[5];
        m_adj[1] = m[2] * m[7] - m[1] * m[8];
        m_adj[2] = m[1] * m[5] - m[2] * m[4];
        m_adj[3] = m[5] * m[6] - m[3] * m[8];
        m_adj[4] = m[0] * m[8] - m[2] * m[6];
        m_adj[5] = m[3] * m[2] - m[0] * m[5];
        m_adj[6] = m[3] * m[7] - m[6] * m[4];
        m_adj[7] = m[6] * m[1] - m[0] * m[7];
        m_adj[8] = m[0] * m[4] - m[3] * m[1];

        det = m[0] * m_adj[0] + m[1] * m_adj[3] + m[2] * m_adj[6];
        if (abs(det) < eps_det) {
            is_bad_face = true;
            return;
        }
        det_inv = 1. / det;
        for (int i = 0; i < 9; ++i) {
            m_adj[i] *= det_inv;
            m_adj_scalar_t[i] = m_adj[i];
        }

        col_sum_x = m_adj[0] + m_adj[3] + m_adj[6];
        col_sum_y = m_adj[1] + m_adj[4] + m_adj[7];
        col_sum_z = m_adj[2] + m_adj[5] + m_adj[8];
        if (abs(col_sum_z) < eps) {
            is_bad_face = true;
            return;
        }
        col_sum_z_inv = 1. / col_sum_z;
    } else if (threadIdx.x == 1) {
        min_x = min(min(m[0], m[1]), m[2]);
        max_x = max(max(m[0], m[1]), m[2]);

        min_y = min(min(m[3], m[4]), m[5]);
        max_y = max(max(m[3], m[4]), m[5]);

        left = lower_bound<scalar_t>(xs_sort, min_x - eps, vertices_filter.size(0));
        right = lower_bound<scalar_t>(xs_sort, max_x + eps, vertices_filter.size(0));
        if (left >= right) {
            is_bad_face = true;
            return;
        }
        vertices_per_thread = (right - left) / blockDim.x + 1;
    }
    __syncthreads();
    if (is_bad_face) {
        return;
    }

    const int start = left + vertices_per_thread * threadIdx.x;
    const int end = min(start + vertices_per_thread, right);

    scalar_t x, y, z, face_z, wa, wb, wc;
    int vertice_index;

    for (int i = start; i < end; i++) {
        vertice_index = indexes[i];

        // vertice already filtered out
        if (vertices_filter[vertice_index] == 0) {
            continue;
        }

        if (vertice_index == ai || vertice_index == bi || vertice_index == ci) {
            return;
        }

        z = zs[vertice_index];
        if (z < eps) {
            continue;
        }
        x = xs[vertice_index];
        y = ys[vertice_index];

        if (x < min_x - eps_border || x > max_x + eps_border || y < min_y - eps_border || y > max_y + eps_border) {
        // if (y < min_y - eps_border || y > max_y + eps_border) {
            continue;
        }

        face_z = col_sum_z_inv * (1 - x * col_sum_x - y * col_sum_y);

        // face in front of vertice
        if (face_z < z) {
            // barycentric coordinates
            wa = x * m_adj_scalar_t[0] + y * m_adj_scalar_t[1] + face_z * m_adj_scalar_t[2];
            if (wa < eps_border_left_0 || wa > eps_border_right_1) {
                continue;
            }
            wb = x * m_adj_scalar_t[3] + y * m_adj_scalar_t[4] + face_z * m_adj_scalar_t[5];
            if (wb < eps_border_left_0 || wb > eps_border_right_1) {
                continue;
            }
            wc = x * m_adj_scalar_t[6] + y * m_adj_scalar_t[7] + face_z * m_adj_scalar_t[8];
            if (wc < eps_border_left_0 || wc > eps_border_right_1) {
                continue;
            }
            // 2d vertice inside 2d triangle
            vertices_filter[vertice_index] = 0;
        }
    }
}


template <typename scalar_t>
__global__ void project_mesh_cuda_kernel_0(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
    scalar_t* xs, scalar_t* ys, scalar_t* zs,
    scalar_t fx, scalar_t fy,
    scalar_t cx, scalar_t cy,
    int H, int W
) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertices.size(0)) {
        return;
    }

    const scalar_t eps = 1e-6;
    scalar_t z = vertices[i][2];
    if (abs(z) < eps) {
        z = -1;
    }
    const scalar_t z_inv = 1. / z;
    xs[i] = vertices[i][0] * z_inv * fx + cx;
    xs[i] /= W;
    ys[i] = vertices[i][1] * z_inv * fy + cy;
    ys[i] /= H;
    zs[i] = z;
}


template <typename scalar_t>
__global__ void project_mesh_cuda_kernel_1(
     const scalar_t* xs, const scalar_t* ys, const scalar_t* zs,
     const torch::PackedTensorAccessor32<int32_t,2> faces,
     const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
     torch::PackedTensorAccessor32<scalar_t,2> depth,
     scalar_t* global_face_inv,
     int* global_is_bad_face
) {
    const int face_indx = blockIdx.x;
    const int H = depth.size(0);
    const int W = depth.size(1);

    scalar_t min_x, max_x, min_y, max_y;
    scalar_t denom;

    __shared__ int vertices_per_thread_x, vertices_per_thread_y;
    __shared__ int ai, bi, ci;
    __shared__ bool is_bad_face;
    __shared__ int min_xi, max_xi, min_yi, max_yi;
    __shared__ scalar_t face[9]; // cartesian coordinates with z untouched
    __shared__ scalar_t face_inv[9];
    const scalar_t eps = 1e-5;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        if (vertices_filter[ai] == 0 || vertices_filter[bi] == 0 || vertices_filter[ci] == 0) {
            is_bad_face = true;
            global_is_bad_face[face_indx] = 1;
            return;
        }

        face[0] = xs[ai]; face[1] = ys[ai]; face[2] = zs[ai];
        face[3] = xs[bi]; face[4] = ys[bi]; face[5] = zs[bi];
        face[6] = xs[ci]; face[7] = ys[ci]; face[8] = zs[ci];

        // negative vertex
        is_bad_face = false;
        if (face[2] < eps or face[5] < eps or face[8] < eps) {
            is_bad_face = true;
            global_is_bad_face[face_indx] = 1;
            return;
        }

        face_inv[0] = face[4] - face[7];
        face_inv[1] = face[6] - face[3];
        face_inv[2] = face[3] * face[7] - face[6] * face[4];
        face_inv[3] = face[7] - face[1];
        face_inv[4] = face[0] - face[6];
        face_inv[5] = face[6] * face[1] - face[0] * face[7];
        face_inv[6] = face[1] - face[4];
        face_inv[7] = face[3] - face[0];
        face_inv[8] = face[0] * face[4] - face[3] * face[1];

        denom = (
            face[6] * (face[1] - face[4]) +
            face[0] * (face[4] - face[7]) +
            face[3] * (face[7] - face[1])
        );

//        if (abs(denom) < eps) {
//            is_bad_face = true;
//            global_is_bad_face[face_indx] = 1;
//            return;
//        }

        for (int i = 0; i < 9; ++i) {
            face_inv[i] /= denom;
        }

        for (int i = 0; i < 9; ++i) {
            global_face_inv[9 * face_indx + i] = face_inv[i];
        }

        global_is_bad_face[face_indx] = 0;

        min_x = min(min(face[0], face[3]), face[6]) * W;
        min_xi = static_cast<int>(floorf(static_cast<float>(min_x)));
        min_xi = min(max(min_xi, 0), W - 1);
        max_x = max(max(face[0], face[3]), face[6]) * W;
        max_xi = static_cast<int>(ceilf(static_cast<float>(max_x)));
        max_xi = min(max(max_xi, 0), W - 1);

        min_y = min(min(face[1], face[4]), face[7]) * H;
        min_yi = static_cast<int>(floorf(static_cast<float>(min_y)));
        min_yi = min(max(min_yi, 0), H - 1);
        max_y = max(max(face[1], face[4]), face[7]) * H;
        max_yi = static_cast<int>(ceilf(static_cast<float>(max_y)));
        max_yi = min(max(max_yi, 0), H - 1);

        vertices_per_thread_x = (max_xi - min_xi) / blockDim.x + 1;
        vertices_per_thread_y = (max_yi - min_yi) / blockDim.y + 1;
    }
    __syncthreads();
    if (is_bad_face) {
        return;
    }

    const int left = min_xi + vertices_per_thread_x * threadIdx.x;
    const int right = min(left + vertices_per_thread_x, max_xi);

    const int top = min_yi + vertices_per_thread_y * threadIdx.y;
    const int bottom = min(top + vertices_per_thread_y, max_yi);

    scalar_t x, y, face_z, wa, wb, wc, wsum;
    for (int i = top; i < bottom; i++) {
        for (int j = left; j < right; j++) {
            x = (scalar_t)j / W;
            y = (scalar_t)i / H;

            // check pixel is inside the face
            if (((y - face[1]) * (face[3] - face[0]) > (x - face[0]) * (face[4] - face[1])) ||
                ((y - face[4]) * (face[6] - face[3]) > (x - face[3]) * (face[7] - face[4])) ||
                ((y - face[7]) * (face[0] - face[6]) > (x - face[6]) * (face[1] - face[7]))) {
                continue;
            }

            wa = face_inv[0] * x + face_inv[1] * y + face_inv[2];
            wb = face_inv[3] * x + face_inv[4] * y + face_inv[5];
            wc = face_inv[6] * x + face_inv[7] * y + face_inv[8];

            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            face_z = 1. / (wa / face[2] + wb / face[5] + wc / face[8]);

            if (sizeof(scalar_t) == sizeof(double)) {
                atomicMin_double((double*)&depth[i][j], (double)face_z);
            } else {
                atomicMinFloat(&depth[i][j], face_z);
            }
        }
    }
}


template <typename scalar_t>
__global__ void project_mesh_cuda_kernel_2(
     const scalar_t* xs, const scalar_t* ys, const scalar_t* zs,
     const torch::PackedTensorAccessor32<int32_t,2> faces,
     const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
     const torch::PackedTensorAccessor32<scalar_t,2> depth,
     const scalar_t* global_face_inv,
     const int* global_is_bad_face,
     const torch::PackedTensorAccessor32<scalar_t,2> vertice_values,
     torch::PackedTensorAccessor32<scalar_t,3> result
) {
    const int face_indx = blockIdx.x;

    if (global_is_bad_face[face_indx]) {
        return;
    }

    const int H = depth.size(0);
    const int W = depth.size(1);
    const int C = vertice_values.size(1);
    const scalar_t eps = 1e-5;

    scalar_t min_x, max_x, min_y, max_y;
    __shared__ int vertices_per_thread_x, vertices_per_thread_y;
    __shared__ int ai, bi, ci;
    __shared__ scalar_t face[9]; // cartesian coordinates with z untouched
    __shared__ scalar_t face_inv[9];
    __shared__ int min_xi, max_xi, min_yi, max_yi;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        face[0] = xs[ai]; face[1] = ys[ai]; face[2] = zs[ai];
        face[3] = xs[bi]; face[4] = ys[bi]; face[5] = zs[bi];
        face[6] = xs[ci]; face[7] = ys[ci]; face[8] = zs[ci];

        for (int i = 0; i < 9; ++i) {
            face_inv[i] = global_face_inv[9 * face_indx + i];
        }

        min_x = min(min(face[0], face[3]), face[6]) * W;
        min_xi = static_cast<int>(floorf(static_cast<float>(min_x)));
        min_xi = min(max(min_xi, 0), W - 1);
        max_x = max(max(face[0], face[3]), face[6]) * W;
        max_xi = static_cast<int>(ceilf(static_cast<float>(max_x)));
        max_xi = min(max(max_xi, 0), W - 1);

        min_y = min(min(face[1], face[4]), face[7]) * H;
        min_yi = static_cast<int>(floorf(static_cast<float>(min_y)));
        min_yi = min(max(min_yi, 0), H - 1);
        max_y = max(max(face[1], face[4]), face[7]) * H;
        max_yi = static_cast<int>(ceilf(static_cast<float>(max_y)));
        max_yi = min(max(max_yi, 0), H - 1);

        vertices_per_thread_x = (max_xi - min_xi) / blockDim.x + 1;
        vertices_per_thread_y = (max_yi - min_yi) / blockDim.y + 1;
    }
    __syncthreads();

    const int left = min_xi + vertices_per_thread_x * threadIdx.x;
    const int right = min(left + vertices_per_thread_x, max_xi);

    const int top = min_yi + vertices_per_thread_y * threadIdx.y;
    const int bottom = min(top + vertices_per_thread_y, max_yi);

    scalar_t x, y, face_z, wa, wb, wc, wsum;
    for (int i = top; i < bottom; i++) {
        for (int j = left; j < right; j++) {
            x = (scalar_t)j / W;
            y = (scalar_t)i / H;

            // check pixel is inside the face
            if (((y - face[1]) * (face[3] - face[0]) > (x - face[0]) * (face[4] - face[1])) ||
                ((y - face[4]) * (face[6] - face[3]) > (x - face[3]) * (face[7] - face[4])) ||
                ((y - face[7]) * (face[0] - face[6]) > (x - face[6]) * (face[1] - face[7]))) {
                continue;
            }

            wa = face_inv[0] * x + face_inv[1] * y + face_inv[2];
            wb = face_inv[3] * x + face_inv[4] * y + face_inv[5];
            wc = face_inv[6] * x + face_inv[7] * y + face_inv[8];

            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            face_z = 1. / (wa / face[2] + wb / face[5] + wc / face[8]);

            if (face_z - eps < depth[i][j]) {
                wsum = wa + wb + wc;
                wa /= wsum; wb /= wsum; wc /= wsum;

                for (int c = 0; c < C; c++) {
                    result[i][j][c] = wa * vertice_values[ai][c] + wb * vertice_values[bi][c] + wc * vertice_values[ci][c];
                }
            }
        }
    }
}


template <typename scalar_t>
__global__ void estimate_normals_cuda_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    const scalar_t* xs, const scalar_t* ys, const scalar_t* zs,
    const torch::PackedTensorAccessor32<int32_t,2> faces,
    const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
    const torch::PackedTensorAccessor32<scalar_t,2> depth,
    const scalar_t* global_face_inv,
    const int* global_is_bad_face,
    torch::PackedTensorAccessor32<scalar_t,3> coords,
    torch::PackedTensorAccessor32<scalar_t,3> normals
) {
    const int face_indx = blockIdx.x;

    if (global_is_bad_face[face_indx]) {
        return;
    }

    const int H = depth.size(0);
    const int W = depth.size(1);
    const scalar_t eps = 1e-5;

    scalar_t min_x, max_x, min_y, max_y;
    scalar_t v1x, v1y, v1z, v2x, v2y, v2z, nlen;
    __shared__ int vertices_per_thread_x, vertices_per_thread_y;
    __shared__ int ai, bi, ci;
    __shared__ scalar_t face_homo[9]; // homogeneous coordinates
    __shared__ scalar_t face_cart[9]; // cartesian coordinates with z untouched
    __shared__ scalar_t face_inv[9];
    __shared__ int min_xi, max_xi, min_yi, max_yi;
    __shared__ scalar_t nx, ny, nz;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        ai = faces[face_indx][0];
        bi = faces[face_indx][1];
        ci = faces[face_indx][2];

        face_cart[0] = xs[ai]; face_cart[1] = ys[ai]; face_cart[2] = zs[ai];
        face_cart[3] = xs[bi]; face_cart[4] = ys[bi]; face_cart[5] = zs[bi];
        face_cart[6] = xs[ci]; face_cart[7] = ys[ci]; face_cart[8] = zs[ci];

        face_homo[0] = vertices[ai][0]; face_homo[1] = vertices[ai][1]; face_homo[2] = vertices[ai][2];
        face_homo[3] = vertices[bi][0]; face_homo[4] = vertices[bi][1]; face_homo[5] = vertices[bi][2];
        face_homo[6] = vertices[ci][0]; face_homo[7] = vertices[ci][1]; face_homo[8] = vertices[ci][2];

        v1x = face_homo[3] - face_homo[0]; v2x = face_homo[6] - face_homo[0];
        v1y = face_homo[4] - face_homo[1]; v2y = face_homo[7] - face_homo[1];
        v1z = face_homo[5] - face_homo[2]; v2z = face_homo[8] - face_homo[2];

        nx = v1y * v2z - v1z * v2y;
        ny = v1z * v2x - v1x * v2z;
        nz = v1x * v2y - v1y * v2x;
        nlen = nx * nx + ny * ny + nz * nz;
        nlen = (scalar_t)sqrt((float)nlen);
        nx /= nlen;
        ny /= nlen;
        nz /= nlen;

        for (int i = 0; i < 9; ++i) {
            face_inv[i] = global_face_inv[9 * face_indx + i];
        }

        min_x = min(min(face_cart[0], face_cart[3]), face_cart[6]) * W;
        min_xi = static_cast<int>(floorf(static_cast<float>(min_x)));
        min_xi = min(max(min_xi, 0), W - 1);
        max_x = max(max(face_cart[0], face_cart[3]), face_cart[6]) * W;
        max_xi = static_cast<int>(ceilf(static_cast<float>(max_x)));
        max_xi = min(max(max_xi, 0), W - 1);

        min_y = min(min(face_cart[1], face_cart[4]), face_cart[7]) * H;
        min_yi = static_cast<int>(floorf(static_cast<float>(min_y)));
        min_yi = min(max(min_yi, 0), H - 1);
        max_y = max(max(face_cart[1], face_cart[4]), face_cart[7]) * H;
        max_yi = static_cast<int>(ceilf(static_cast<float>(max_y)));
        max_yi = min(max(max_yi, 0), H - 1);

        vertices_per_thread_x = (max_xi - min_xi) / blockDim.x + 1;
        vertices_per_thread_y = (max_yi - min_yi) / blockDim.y + 1;
    }
    __syncthreads();

    const int left = min_xi + vertices_per_thread_x * threadIdx.x;
    const int right = min(left + vertices_per_thread_x, max_xi);

    const int top = min_yi + vertices_per_thread_y * threadIdx.y;
    const int bottom = min(top + vertices_per_thread_y, max_yi);

    scalar_t x, y, face_z, wa, wb, wc, wsum;
    for (int i = top; i < bottom; i++) {
        for (int j = left; j < right; j++) {
            x = (scalar_t)j / W;
            y = (scalar_t)i / H;

            // check pixel is inside the face
            if (((y - face_cart[1]) * (face_cart[3] - face_cart[0]) > (x - face_cart[0]) * (face_cart[4] - face_cart[1])) ||
                ((y - face_cart[4]) * (face_cart[6] - face_cart[3]) > (x - face_cart[3]) * (face_cart[7] - face_cart[4])) ||
                ((y - face_cart[7]) * (face_cart[0] - face_cart[6]) > (x - face_cart[6]) * (face_cart[1] - face_cart[7]))) {
                continue;
            }

            wa = face_inv[0] * x + face_inv[1] * y + face_inv[2];
            wb = face_inv[3] * x + face_inv[4] * y + face_inv[5];
            wc = face_inv[6] * x + face_inv[7] * y + face_inv[8];

            wsum = wa + wb + wc;
            wa /= wsum; wb /= wsum; wc /= wsum;

            face_z = 1. / (wa / face_cart[2] + wb / face_cart[5] + wc / face_cart[8]);

            if (face_z - eps < depth[i][j]) {
                coords[i][j][0] = wa * face_homo[0] + wb * face_homo[3] + wc * face_homo[6];
                coords[i][j][1] = wa * face_homo[1] + wb * face_homo[4] + wc * face_homo[7];
                coords[i][j][2] = wa * face_homo[2] + wb * face_homo[5] + wc * face_homo[8];

                normals[i][j][0] = nx;
                normals[i][j][1] = ny;
                normals[i][j][2] = nz;
            }
        }
    }
}

template <typename scalar_t>
__global__ void project_vertices_2d_cuda_kernel_0(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
    torch::PackedTensorAccessor32<int32_t,2> vertice_indxs,
    torch::PackedTensorAccessor32<scalar_t,2> depth,
    scalar_t fx, scalar_t fy,
    scalar_t cx, scalar_t cy
) {
    const scalar_t eps = 1e-6;
    const int H = vertice_indxs.size(0);
    const int W = vertice_indxs.size(1);
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= vertices.size(0)) {
        return;
    }

    if (vertices_filter[i] == 0) {
        return;
    }

    scalar_t x, y, z, z_global;
    x = vertices[i][0];
    y = vertices[i][1];
    z = vertices[i][2];
    if (z < eps) {
        return;
    }

    x = x / z * fx + cx;
    y = y / z * fy + cy;

    // round to nearest
    const int xi = static_cast<int>(floorf(static_cast<float>(x) + 0.5));
    const int yi = static_cast<int>(floorf(static_cast<float>(y) + 0.5));

    if (0 <= xi && xi < W && 0 <= yi && yi < H) {
        z_global = depth[yi][xi];
        if (z_global < 0 || z < z_global) {
            depth[yi][xi] = z;
            vertice_indxs[yi][xi] = i;
        }
    }
}


template <typename scalar_t>
__global__ void project_vertices_2d_cuda_kernel_1(
    const torch::PackedTensorAccessor32<scalar_t,2> vertice_values,
    const torch::PackedTensorAccessor32<int32_t,2> vertice_indxs,
    torch::PackedTensorAccessor32<scalar_t,3> result
) {
    const int C = vertice_values.size(1);
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;
    if (xi >= result.size(1) || yi >= result.size(0)) {
        return;
    }
    const int vertice_index = vertice_indxs[yi][xi];
    if (vertice_index >= 0) {
        for (int c = 0; c < C; c++) {
            result[yi][xi][c] = vertice_values[vertice_index][c];
        }
    }
}


template <typename scalar_t>
__global__ void project_vertices_3d_cuda_kernel_0(
    const torch::PackedTensorAccessor32<scalar_t,2> vertices,
    const torch::PackedTensorAccessor32<uint8_t,1> vertices_filter,
    torch::PackedTensorAccessor32<int32_t,3> vertice_indxs,
    torch::PackedTensorAccessor32<scalar_t,3> dists,
    scalar_t fx, scalar_t fy, scalar_t fz,
    scalar_t cx, scalar_t cy, scalar_t cz
) {
    const int dimz = vertice_indxs.size(0);
    const int dimy = vertice_indxs.size(1);
    const int dimx = vertice_indxs.size(2);
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= vertices.size(0)) {
        return;
    }

    if (vertices_filter[i] == 0) {
        return;
    }

    scalar_t x, y, z, dist, dist_global;
    x = vertices[i][0];
    y = vertices[i][1];
    z = vertices[i][2];

    x = x * fx + cx;
    y = y * fy + cy;
    z = z * fz + cz;

    // round to nearest
    const int xi = static_cast<int>(floorf(static_cast<float>(x) + 0.5));
    const int yi = static_cast<int>(floorf(static_cast<float>(y) + 0.5));
    const int zi = static_cast<int>(floorf(static_cast<float>(z) + 0.5));

    if (0 <= xi && xi < dimx && 0 <= yi && yi < dimy && 0 <= zi && zi < dimz) {
        dist = abs(x - (scalar_t)xi) + abs(y - (scalar_t)yi) + abs(z - (scalar_t)zi);
        dist_global = dists[zi][yi][xi];
        if (dist_global < 0 || dist < dist_global) {
            dists[zi][yi][xi] = dist;
            vertice_indxs[zi][yi][xi] = i;
        }
    }
}

template <typename scalar_t>
__global__ void project_vertices_3d_cuda_kernel_1(
    const torch::PackedTensorAccessor32<scalar_t,2> vertice_values,
    const torch::PackedTensorAccessor32<int32_t,3> vertice_indxs,
    torch::PackedTensorAccessor32<scalar_t,4> result
) {
    const int C = vertice_values.size(1);
    const int xi = threadIdx.x + blockIdx.x * blockDim.x;
    const int yi = threadIdx.y + blockIdx.y * blockDim.y;
    const int zi = threadIdx.z + blockIdx.z * blockDim.z;
    if (xi >= result.size(2) || yi >= result.size(1) || zi >= result.size(0)) {
        return;
    }
    const int vertice_index = vertice_indxs[zi][yi][xi];
    if (vertice_index >= 0) {
        for (int c = 0; c < C; c++) {
            result[zi][yi][xi][c] = vertice_values[vertice_index][c];
        }
    }
}

// cpp defined functions

torch::Tensor z_filter_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces
) {
    const int gpuid = vertices.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));

    const int M = faces.size(0);
    const int N = vertices.size(0);

    const dim3 dimBlock0(BLOCK_SIZE);
    const dim3 dimGrid0((N - 1) / dimBlock0.x + 1);

    const dim3 dimGrid1(M);
    const dim3 dimBlock1(16);
    auto vertices_filter = torch::ones({N}, torch::dtype(torch::kUInt8).device(torch::kCUDA, gpuid));
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "z_filter_cuda_kernel", [&] {
        scalar_t* xs;
        scalar_t* ys;
        scalar_t* zs;
        cudaMalloc(&xs, N * sizeof(scalar_t));
        cudaMalloc(&ys, N * sizeof(scalar_t));
        cudaMalloc(&zs, N * sizeof(scalar_t));

        int* indexes;
        cudaMalloc(&indexes, N * sizeof(int));

        z_filter_cuda_kernel_0<scalar_t><<<dimGrid0, dimBlock0>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            xs, ys, zs,
            indexes
        );
        AT_CUDA_CHECK(cudaGetLastError());

        scalar_t* xs_sort;
        cudaMalloc(&xs_sort, N * sizeof(scalar_t));
        cudaMemcpy(xs_sort, xs, N * sizeof(scalar_t), cudaMemcpyDeviceToDevice);
        thrust::device_ptr<scalar_t> xs_sort_thrust = thrust::device_pointer_cast(xs_sort);
        thrust::device_ptr<int> indexes_thrust = thrust::device_pointer_cast(indexes);
        thrust::sort_by_key(xs_sort_thrust, xs_sort_thrust + N, indexes_thrust);
        AT_CUDA_CHECK(cudaGetLastError());

        z_filter_cuda_kernel_1<scalar_t><<<dimGrid1, dimBlock1>>>(
            xs, ys, zs,
            xs_sort, indexes,
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>()
        );
        AT_CUDA_CHECK(cudaGetLastError());

        cudaFree(xs);
        cudaFree(ys);
        cudaFree(zs);
        cudaFree(indexes);
        cudaFree(xs_sort);
        AT_CUDA_CHECK(cudaGetLastError());
    });

    return vertices_filter;
}

std::vector<torch::Tensor> estimate_normals_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int H, int W
) {
    const int N = vertices.size(0);
    const int M = faces.size(0);

    const int gpuid = vertices.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(vertices.scalar_type()).device(torch::kCUDA, gpuid);

    const dim3 dimBlock0(BLOCK_SIZE);
    const dim3 dimGrid0((N - 1) / dimBlock0.x + 1);

    const dim3 dimGrid1(M);
    const dim3 dimBlock1(4, 4);

    auto depth = torch::ones({H, W}, options) * 1e10;
    auto coords = torch::zeros({H, W, 3}, options);
    auto normals = torch::zeros({H, W, 3}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "project_mesh_cuda_kernel", [&] {
        scalar_t* xs;
        scalar_t* ys;
        scalar_t* zs;
        cudaMalloc(&xs, N * sizeof(scalar_t));
        cudaMalloc(&ys, N * sizeof(scalar_t));
        cudaMalloc(&zs, N * sizeof(scalar_t));

        project_mesh_cuda_kernel_0<scalar_t><<<dimGrid0, dimBlock0>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            xs, ys, zs,
            fx, fy,
            cx, cy,
            H, W
        );
        AT_CUDA_CHECK(cudaGetLastError());

        scalar_t* global_face_inv;
        cudaMalloc(&global_face_inv, M * 9 * sizeof(scalar_t));
        int* global_is_bad_face;
        cudaMalloc(&global_is_bad_face, M * sizeof(int));
        project_mesh_cuda_kernel_1<scalar_t><<<dimGrid1, dimBlock1>>>(
            xs, ys, zs,
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_inv,
            global_is_bad_face
        );
        AT_CUDA_CHECK(cudaGetLastError());

        estimate_normals_cuda_kernel<scalar_t><<<dimGrid1, dimBlock1>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            xs, ys, zs,
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_inv,
            global_is_bad_face,
            coords.packed_accessor32<scalar_t,3>(),
            normals.packed_accessor32<scalar_t,3>()
        );
        AT_CUDA_CHECK(cudaGetLastError());

        cudaFree(xs);
        cudaFree(ys);
        cudaFree(zs);
        cudaFree(global_face_inv);
        cudaFree(global_is_bad_face);
        AT_CUDA_CHECK(cudaGetLastError());
    });

    return {coords, normals};
}

torch::Tensor project_mesh_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int H, int W
) {
    const int N = vertices.size(0);
    const int C = vertice_values.size(1);
    const int M = faces.size(0);

    const int gpuid = vertices.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(vertices.scalar_type()).device(torch::kCUDA, gpuid);

    const dim3 dimBlock0(BLOCK_SIZE);
    const dim3 dimGrid0((N - 1) / dimBlock0.x + 1);

    const dim3 dimGrid1(M);
    const dim3 dimBlock1(4, 4);

    auto depth = torch::ones({H, W}, options) * 1e10;
    auto result = torch::zeros({H, W, C}, options);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "project_mesh_cuda_kernel", [&] {
        scalar_t* xs;
        scalar_t* ys;
        scalar_t* zs;
        cudaMalloc(&xs, N * sizeof(scalar_t));
        cudaMalloc(&ys, N * sizeof(scalar_t));
        cudaMalloc(&zs, N * sizeof(scalar_t));

        project_mesh_cuda_kernel_0<scalar_t><<<dimGrid0, dimBlock0>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            xs, ys, zs,
            fx, fy,
            cx, cy,
            H, W
        );
        AT_CUDA_CHECK(cudaGetLastError());

        scalar_t* global_face_inv;
        cudaMalloc(&global_face_inv, M * 9 * sizeof(scalar_t));
        int* global_is_bad_face;
        cudaMalloc(&global_is_bad_face, M * sizeof(int));
        project_mesh_cuda_kernel_1<scalar_t><<<dimGrid1, dimBlock1>>>(
            xs, ys, zs,
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_inv,
            global_is_bad_face
        );
        AT_CUDA_CHECK(cudaGetLastError());

        project_mesh_cuda_kernel_2<scalar_t><<<dimGrid1, dimBlock1>>>(
            xs, ys, zs,
            faces.packed_accessor32<int32_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            depth.packed_accessor32<scalar_t,2>(),
            global_face_inv,
            global_is_bad_face,
            vertice_values.packed_accessor32<scalar_t,2>(),
            result.packed_accessor32<scalar_t,3>()
        );
        AT_CUDA_CHECK(cudaGetLastError());

        cudaFree(xs);
        cudaFree(ys);
        cudaFree(zs);
        cudaFree(global_face_inv);
        cudaFree(global_is_bad_face);
        AT_CUDA_CHECK(cudaGetLastError());
    });

    return result;
}

torch::Tensor project_vertices_2d_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int H, int W
) {
    const int N = vertices.size(0);
    const int C = vertice_values.size(1);

    const int gpuid = vertices.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(vertices.scalar_type()).device(torch::kCUDA, gpuid);
    auto depth = torch::ones({H, W}, options) * -1;
    auto vertice_indxs = torch::ones(
        {H, W},
        torch::dtype(torch::kInt32).device(torch::kCUDA, gpuid)
    ) * -1;

    const dim3 dimBlock0(BLOCK_SIZE);
    const dim3 dimGrid0((N - 1) / dimBlock0.x + 1);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "project_vertices_2d_cuda_kernel_0", [&] {
        project_vertices_2d_cuda_kernel_0<scalar_t><<<dimGrid0, dimBlock0>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            vertice_indxs.packed_accessor32<int32_t,2>(),
            depth.packed_accessor32<scalar_t,2>(),
            fx, fy,
            cx, cy
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    auto result = torch::zeros({H, W, C}, options);
    const dim3 dimBlock1(BLOCK_SIZE_2D_X, BLOCK_SIZE_2D_Y);
    const dim3 dimGrid1((W - 1) / dimBlock1.x + 1, (H - 1) / dimBlock1.y + 1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "project_vertices_2d_cuda_kernel_1", [&] {
        project_vertices_2d_cuda_kernel_1<scalar_t><<<dimGrid1, dimBlock1>>>(
            vertice_values.packed_accessor32<scalar_t,2>(),
            vertice_indxs.packed_accessor32<int32_t,2>(),
            result.packed_accessor32<scalar_t,3>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return result;
}

torch::Tensor project_vertices_3d_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    float fx, float fy, float fz,
    float cx, float cy, float cz,
    int dimx, int dimy, int dimz
) {
    const int gpuid = vertices.device().index();
    AT_CUDA_CHECK(cudaSetDevice(gpuid));
    auto options = torch::dtype(vertices.scalar_type()).device(torch::kCUDA, gpuid);
    const int N = vertices.size(0);
    const int C = vertice_values.size(1);
    auto dists = torch::ones({dimz, dimy, dimx}, options) * -1;
    auto vertice_indxs = torch::ones(
        {dimz, dimy, dimx},
        torch::dtype(torch::kInt32).device(torch::kCUDA, gpuid)
    ) * -1;
    const dim3 dimBlock0(BLOCK_SIZE);
    const dim3 dimGrid0((N - 1) / dimBlock0.x + 1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "project_vertices_3d_cuda_kernel_0", [&] {
        project_vertices_3d_cuda_kernel_0<scalar_t><<<dimGrid0, dimBlock0>>>(
            vertices.packed_accessor32<scalar_t,2>(),
            vertices_filter.packed_accessor32<uint8_t,1>(),
            vertice_indxs.packed_accessor32<int32_t,3>(),
            dists.packed_accessor32<scalar_t,3>(),
            fx, fy, fz,
            cx, cy, cz
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    auto result = torch::zeros({dimz, dimy, dimx, C}, options);
    const dim3 dimBlock1(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    const dim3 dimGrid1((dimx - 1) / dimBlock1.x + 1, (dimy - 1) / dimBlock1.y + 1, (dimz - 1) / dimBlock1.z + 1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vertices.scalar_type(), "project_vertices_3d_cuda_kernel_1", [&] {
        project_vertices_3d_cuda_kernel_1<scalar_t><<<dimGrid1, dimBlock1>>>(
            vertice_values.packed_accessor32<scalar_t,2>(),
            vertice_indxs.packed_accessor32<int32_t,3>(),
            result.packed_accessor32<scalar_t,4>()
        );
    });
    AT_CUDA_CHECK(cudaGetLastError());

    return result;
}