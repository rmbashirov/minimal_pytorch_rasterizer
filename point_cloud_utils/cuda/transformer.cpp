#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor apply_transform_cuda(
    const torch::Tensor& vertices,
    float R[9], float t[3],
    float z_mean, float alpha, bool scale_z_only
);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor apply_transform(
    const torch::Tensor& vertices,
    py::list R_list, py::list t_list,
    float z_mean, float alpha, bool scale_z_only
) {
    CHECK_INPUT(vertices);

    TORCH_CHECK(R_list.size() == 9, "R len should be 9");
    float R[9];
    for (size_t i = 0; i < 9; ++i) {
        R[i] = R_list[i].cast<float>();
    }

    TORCH_CHECK(t_list.size() == 3, "t len should be 3");
    float t[3];
    for (size_t i = 0; i < 3; ++i) {
        t[i] = t_list[i].cast<float>();
    }

    return apply_transform_cuda(
        vertices,
        R, t,
        z_mean, alpha, scale_z_only
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_transform", &apply_transform, "apply_transform (CUDA)");
}
