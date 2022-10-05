from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == '__main__':
    setup(
        name='mod_dcn_op_v2',
        ext_modules=[
            CUDAExtension(
                'mod_dcn_op_v2',
                sources=['src/modulated_deform_conv.cpp', 'src/modulated_deform_conv_cuda.cu'],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
