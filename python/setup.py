import os
import setuptools
import subprocess
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_file_path = os.path.abspath(__file__)
current_file_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(current_file_directory)

thunderkittens_root = os.path.join(project_root, "submodules/ThunderKittens")
python_include = subprocess.check_output(['python', '-c', "import sysconfig; print(sysconfig.get_path('include'))"]).decode().strip()
torch_include = subprocess.check_output(['python', '-c', "import torch; from torch.utils.cpp_extension import include_paths; print(' '.join(['-I' + p for p in include_paths()]))"]).decode().strip()


# CUDA flags
cuda_flags = [
    '-DNDEBUG',
    '-Xcompiler=-Wno-psabi',
    '-Xcompiler=-fno-strict-aliasing',
    '--expt-extended-lambda',
    '--expt-relaxed-constexpr',
    '-forward-unknown-to-host-compiler',
    '--use_fast_math',
    '-std=c++20',
    '-Xnvlink=--verbose',
    '-Xptxas=--verbose',
    '-Xptxas=--warn-on-spills',
    f'-I{python_include}',
    '-DTORCH_COMPILE',
    '-lcuda',
] + torch_include.split()

cuda_flags.append('-DKITTENS_HOPPER')
cuda_flags.append('-arch=sm_90a')


cpp_flags = [
    '-std=c++20',
    '-O3',
    '-lcuda',
]

setuptools.setup(
    name='spattn',
    ext_modules=[
        CUDAExtension(
            name="spattn_interface",
            include_dirs=[
                os.path.join(project_root, "./"),
                os.path.join(thunderkittens_root, "include"),
                os.path.join(thunderkittens_root, "prototype"),
            ],
            sources=[
                "extension.cpp",
                "cuda_attn.cu",
            ],
            extra_compile_args={
                'cxx': cpp_flags, 
                'nvcc': cuda_flags,
            },
            libraries=['cuda'],
        ),
    ],
    # Specify the custom build command for C++/CUDA extensions
    cmdclass={
        'build_ext': BuildExtension
    }
)


 