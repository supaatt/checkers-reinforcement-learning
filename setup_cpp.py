"""
Build script for C++ MCTS module.

Usage:
    pip install pybind11
    python setup_cpp.py build_ext --inplace
"""

import platform
from setuptools import setup, Extension
import pybind11

compile_args = ["-std=c++17", "-O3", "-Wall"]
link_args = []

# macOS needs explicit libc++ linking
if platform.system() == "Darwin":
    compile_args.append("-stdlib=libc++")
    link_args.append("-stdlib=libc++")
    link_args.append("-lc++")

ext = Extension(
    "mcts_cpp",
    sources=["cpp/bindings.cpp"],
    include_dirs=[
        pybind11.get_include(),
        "cpp",
    ],
    language="c++",
    extra_compile_args=compile_args,
    extra_link_args=link_args,
)

setup(
    name="mcts_cpp",
    version="1.0",
    ext_modules=[ext],
    install_requires=["pybind11"],
)