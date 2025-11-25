from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import os
import torch

# --- START: Custom BuildExtension to filter problematic linker flags ---
# We create a custom build class to intercept and modify the compiler arguments.
class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # The original linker arguments might contain a problematic `--sysroot` flag
        # injected by the PyTorch build system.
        original_linker_args = self.compiler.linker_so
        
        # Filter out the specific problematic argument.
        # This is more robust than trying to override it.
        filtered_linker_args = []
        for arg in original_linker_args:
            if arg != '-Wl,--sysroot=/':
                filtered_linker_args.append(arg)
        
        # Replace the compiler's linker arguments with our filtered list.
        self.compiler.linker_so = filtered_linker_args
        
        print("="*50)
        print("Original linker args:", original_linker_args)
        print("Filtered linker args:", self.compiler.linker_so)
        print("="*50)

        # Now, call the original build_extensions method to proceed with the build.
        super().build_extensions()
# --- END: Custom BuildExtension ---


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):
    """
    Helper function to create a CUDA or Cpp extension.
    This version explicitly overrides the linker's sysroot to fix
    compilation issues in certain Conda environments.
    """
    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}
    extra_link_args = []
    
    # This logic is still useful as it provides the CORRECT sysroot path,
    # which we now rely on after removing the incorrect one.
    conda_prefix = os.getenv('CONDA_PREFIX')
    if conda_prefix:
        conda_sysroot = os.path.join(conda_prefix, 'x86_64-conda-linux-gnu', 'sysroot')
        if os.path.isdir(conda_sysroot):
            print(f'Found Conda sysroot: {conda_sysroot}. Overriding linker sysroot.')
            extra_link_args.append(f'--sysroot={conda_sysroot}')
            
            sysroot_lib_path = os.path.join(conda_sysroot, 'lib')
            conda_lib_path = os.path.join(conda_prefix, 'lib')
            
            extra_link_args.append(f'-L{sysroot_lib_path}')
            extra_link_args.append(f'-L{conda_lib_path}')
            extra_link_args.append(f'-Wl,-rpath,{sysroot_lib_path}')
            extra_link_args.append(f'-Wl,-rpath,{conda_lib_path}')

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        print(f'Compiling {name} with CUDA')
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources.extend(sources_cuda)
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    if module == '.':
        ext_name = name
        source_paths = sources
    else:
        ext_name = '{}.{}'.format(module, name)
        source_paths = [os.path.join(*module.split('.'), p) for p in sources]

    return extension(
        name=ext_name,
        sources=source_paths,
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)

def get_ext_modules():
    """
    Collect all extensions to be built.
    """
    return [
        make_cuda_ext(
            name='_msmv_sampling_cuda',
            module='.', 
            sources=[
                'msmv_sampling/msmv_sampling.cpp',
            ],
            sources_cuda=[
                'msmv_sampling/msmv_sampling_forward.cu',
                'msmv_sampling/msmv_sampling_backward.cu'
            ],
            extra_include_path=['msmv_sampling']
        ),
        make_cuda_ext(
            name='bev_pool_v2_ext',
            module='bev_pool_v2',
            sources=[
                'src/bev_pool.cpp'
            ],
            sources_cuda=[
                'src/bev_pool_cuda.cu'
            ]
        ),
    ]

setup(
    name='csrc',
    ext_modules=get_ext_modules(),
    # Use our custom build class instead of the default one.
    cmdclass={'build_ext': CustomBuildExtension}
)

