from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extensions = [
    Extension(
        name="numpoly.cfunctions." + item.replace(".pyx", ""),  # Name of the extension module
        sources=["./numpoly/cfunctions/" + item],  # Path to your `.pyx` file
        include_dirs=[np.get_include()],  # Include NumPy headers
        extra_compile_args=['-fopenmp'], # Enable OpenMP for parallelization
        extra_link_args=['-fopenmp']
    )
    for item in os.listdir("./numpoly/cfunctions/") if ".pyx" in item
]

# Setup configuration
setup(
    ext_modules=cythonize(extensions),  # Compile the Cython files
    include_dirs=[np.get_include()],  # Include NumPy headers
)
