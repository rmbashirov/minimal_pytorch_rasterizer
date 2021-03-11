from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ext_modules = [
    CUDAExtension(
        'minimal_pytorch_rasterizer.cuda.rasterizer', [
            'minimal_pytorch_rasterizer/cuda/rasterizer.cpp',
            'minimal_pytorch_rasterizer/cuda/rasterizer_kernel.cu',
        ])
]

setup(
    version='0.5',
    description='cuda accelerated point cloud utils',
    author='Renat Bashirov',
    author_email='rmbashirov@gmail.com',
    install_requires=["torch>=1.3"],
    packages=['minimal_pytorch_rasterizer', 'minimal_pytorch_rasterizer.cuda'],
    name='minimal_pytorch_rasterizer',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension}
)
