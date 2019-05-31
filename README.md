# GPE-SpectralMethod

## Use

Implementing a spectral method in python for solving the Gross-Pitaevskii Equation in three spatial dimensions. Includes CPU and GPU implementation

The GPU implementation is much faster than the CPU implementation. Depending on the size of the problem, the GPU code can be up to an order of magnitude faster. To use the GPU code, you will need to have a system that has a GPU with CuPy available.

If you don't have a GPU or don't have that installed, the easiest way to use the code is to use google colab (https://colab.research.google.com/), which allows you to use a nice GPU for free.

After opening a Jupyter notebook in colab, you will need to enable the GPU. Go to Runtime > Change Runtime Type and set Hardware Accelerator to GPU to make sure you can use the GPU.

For the code to work, you will also need to add the line

import chainer

to the top of the file so that colab will let you import cupy. Other than that, everything should be the same.

## Notes

GroundStateSave Directory

I am using this directory to save the ground state particular harmonic trap configurations. In the file info.txt, I store information about each ground state to give details about the trapping potential and the dimensions of the region in space over which it spans.

In each file, I can only store the value of the wave function at each grid point and not information about the location at each grid point, so to get around this I am storing that information in info.txt