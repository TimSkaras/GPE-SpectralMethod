# GPE-SpectralMethod

Implementing a spectral method in python for solving the Gross-Pitaevskii Equation in three spatial dimensions. Includes CPU and GPU implementation

The GPU implementation is much faster than the CPU implementation. Depending on the size of the problem, the GPU code can be up to an order of magnitude faster. To use the GPU code, you will need to have a system that has a GPU with CuPy available.

If you don't have a GPU or don't have that installed, the easiest way to use the code is to use google colab (https://colab.research.google.com/), which allows you to use a nice GPU for free.

After opening a Jupyter notebook in colab, you will need to enable the GPU. Go to Runtime > Change Runtime Type and set Hardware Accelerator to GPU to make sure you can use the GPU.

For the code to work, you will also need to add the line

import chainer

to the top of the file so that colab will let you import cupy. Other than that, everything should be the same.