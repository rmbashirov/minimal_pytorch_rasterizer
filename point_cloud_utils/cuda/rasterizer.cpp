#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

torch::Tensor z_filter_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces
);

std::vector<torch::Tensor> estimate_normals_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int h, int w
);


torch::Tensor project_mesh_cuda(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int h, int w
);

torch::Tensor project_vertices_2d_cuda(
    const torch::Tensor& vertices, 
    const torch::Tensor& vertice_values, 
    const torch::Tensor& vertices_filter,
    float fx, float fy, 
    float cx, float cy,
    int h, int w
);

torch::Tensor project_vertices_3d_cuda(
    const torch::Tensor& vertices, 
    const torch::Tensor& vertice_values, 
    const torch::Tensor& vertices_filter,
    float fx, float fy, float fz, 
    float cx, float cy, float cz,
    int dimx, int dimy, int dimz
);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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

torch::Tensor z_filter(
    const torch::Tensor& vertices,
    const torch::Tensor& faces
) {
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    check_equal_gpuid(vertices, faces);
    return z_filter_cuda(vertices, faces);
}

std::vector<torch::Tensor> estimate_normals(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int h, int w
) {
    TORCH_CHECK(h > 0, "h expected to be > 0");
    TORCH_CHECK(w > 0, "w expected to be > 0");
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(vertices_filter);
    return estimate_normals_cuda(
        vertices, faces, vertices_filter,
        fx, fy,
        cx, cy,
        h, w
    );
}

torch::Tensor project_mesh(
    const torch::Tensor& vertices,
    const torch::Tensor& faces,
    const torch::Tensor& vertice_values,
    const torch::Tensor& vertices_filter,
    float fx, float fy,
    float cx, float cy,
    int h, int w
) {
    TORCH_CHECK(h > 0, "h expected to be > 0");
    TORCH_CHECK(w > 0, "w expected to be > 0");
    CHECK_INPUT(vertices);
    CHECK_INPUT(faces);
    CHECK_INPUT(vertice_values);
    CHECK_INPUT(vertices_filter);
    return project_mesh_cuda(
        vertices, faces, vertice_values, vertices_filter,
        fx, fy,
        cx, cy,
        h, w
    );
}

torch::Tensor project_vertices_2d(
    const torch::Tensor& vertices, 
    const torch::Tensor& vertice_values, 
    const torch::Tensor& vertices_filter,
    float fx, float fy, 
    float cx, float cy,
    int h, int w
) {
    TORCH_CHECK(h > 0, "h expected to be > 0");
    TORCH_CHECK(w > 0, "w expected to be > 0");
    CHECK_INPUT(vertices);
    CHECK_INPUT(vertice_values);
    CHECK_INPUT(vertices_filter);
    check_equal_dtype(vertices, vertice_values);
    check_equal_gpuid(vertices, vertice_values);
    check_equal_gpuid(vertice_values, vertices_filter);
    return project_vertices_2d_cuda(
        vertices, vertice_values, vertices_filter,
        fx, fy, 
        cx, cy,
        h, w
    );
}

torch::Tensor project_vertices_3d(
    const torch::Tensor& vertices, 
    const torch::Tensor& vertice_values, 
    const torch::Tensor& vertices_filter,
    float fx, float fy, float fz, 
    float cx, float cy, float cz,
    int dimx, int dimy, int dimz
) {
    TORCH_CHECK(dimx > 0, "dimx expected to be > 0");
    TORCH_CHECK(dimy > 0, "dimy expected to be > 0");
    TORCH_CHECK(dimz > 0, "dimz expected to be > 0");
    CHECK_INPUT(vertices);
    CHECK_INPUT(vertice_values);
    CHECK_INPUT(vertices_filter);
    check_equal_dtype(vertices, vertice_values);
    check_equal_gpuid(vertices, vertice_values);
    check_equal_gpuid(vertice_values, vertices_filter);
    return project_vertices_3d_cuda(
        vertices, vertice_values, vertices_filter,
        fx, fy, fz, 
        cx, cy, cz,
        dimx, dimy, dimz
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("z_filter", &z_filter, "z_filter (CUDA)");
    m.def("estimate_normals", &estimate_normals, "estimate_normals (CUDA)");
    m.def("project_mesh", &project_mesh, "project_mesh (CUDA)");
    m.def("project_vertices_2d", &project_vertices_2d, "project_vertices (CUDA)");
    m.def("project_vertices_3d", &project_vertices_3d, "project_vertices_3d (CUDA)");
}