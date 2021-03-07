from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = [
    CUDAExtension(
        'point_cloud_utils.cuda.loader', [
            'point_cloud_utils/cuda/loader.cpp',
            'point_cloud_utils/cuda/loader_kernel.cu',
        ]),
    CUDAExtension(
        'point_cloud_utils.cuda.transformer', [
            'point_cloud_utils/cuda/transformer.cpp',
            'point_cloud_utils/cuda/transformer_kernel.cu',
        ]),
    CUDAExtension(
        'point_cloud_utils.cuda.rasterizer', [
            'point_cloud_utils/cuda/rasterizer.cpp',
            'point_cloud_utils/cuda/rasterizer_kernel.cu',
        ])
]

setup(
    version='0.4',
    description='cuda accelerated point cloud utils',
    author='Renat Bashirov',
    author_email='rmbashirov@gmail.com',
    install_requires=["torch>=1.3"],
    packages=['point_cloud_utils', 'point_cloud_utils.cuda'],
    name='point_cloud_utils',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
