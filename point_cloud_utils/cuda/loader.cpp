#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_mesh_cuda(
    const torch::Tensor& depth, 
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float near, float far,
    float fx, float fy, 
    float cx, float cy
);


std::tuple<torch::Tensor, torch::Tensor> create_point_cloud_cuda(
    const torch::Tensor& depth, 
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float near, float far,
    float fx, float fy, 
    float cx, float cy
);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void check_shapes(const torch::Tensor& a, const torch::Tensor& b, int dim_a, int dim_b) {
    TORCH_CHECK(a.dim() == dim_a, "expected ", dim_a, " dims but tensor has ", a.dim());
    TORCH_CHECK(b.dim() == dim_b, "expected ", dim_b, " dims but tensor has ", b.dim());
    TORCH_CHECK(a.size(0) == b.size(0), "dim 0 do not match, got ", a.size(0), " != ", b.size(0));
    TORCH_CHECK(a.size(1) == b.size(1), "dim 1 do not match, got ", a.size(1), " != ", b.size(1));
}

void check_equal_dtype(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.dtype() == b.dtype(), 
        "expected equal dtype, got ", a.dtype(), " != ", b.dtype()
    );
}

void check_equal_gpuid(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(
        a.device().index() == b.device().index(), 
        "expected equal gpu id, got ", a.device().index(), " != ", b.device().index()
    );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> create_mesh(
    const torch::Tensor& depth, 
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float near, float far,
    float fx, float fy, 
    float cx, float cy
) {
    CHECK_INPUT(depth);
    CHECK_INPUT(values);
    CHECK_INPUT(mask);
    check_shapes(depth, values, 2, 3);
    check_shapes(depth, mask, 2, 2);
    check_equal_dtype(depth, values);
    check_equal_gpuid(depth, values);
    check_equal_gpuid(values, mask);
    return create_mesh_cuda(
        depth, values, mask,
        near, far, 
        fx, fy, 
        cx, cy
    );
}

std::tuple<torch::Tensor, torch::Tensor> create_point_cloud(
    const torch::Tensor& depth, 
    const torch::Tensor& values,
    const torch::Tensor& mask,
    float near, float far,
    float fx, float fy, 
    float cx, float cy
) {
    CHECK_INPUT(depth);
    CHECK_INPUT(values);
    CHECK_INPUT(mask);
    check_shapes(depth, values, 2, 3);
    check_shapes(depth, mask, 2, 2);
    check_equal_dtype(depth, values);
    check_equal_gpuid(depth, values);
    check_equal_gpuid(values, mask);
    return create_point_cloud_cuda(
        depth, values, mask,
        near, far, 
        fx, fy, 
        cx, cy
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_mesh", &create_mesh, "create_mesh (CUDA)");
    m.def("create_point_cloud", &create_point_cloud, "create_point_cloud (CUDA)");
}
