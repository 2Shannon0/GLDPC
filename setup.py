from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "gldpc_decoder_fix_iter",
        ["gldpc_decoder_fix_iter.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],  # Добавили поддержку C++17
    ),
]

setup(
    name="gldpc_decoder_fix_iter",
    ext_modules=ext_modules,
)
