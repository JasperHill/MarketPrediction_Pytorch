from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='rnn_passes_cpp',
      ext_modules=[cpp_extension.CppExtension('rnn_passes_cpp', ['RNN_passes.cpp'])],
      cmdclass={'build_ext' : cpp_extension.BuildExtension})
