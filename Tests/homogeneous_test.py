import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import timeit
import numba
from numba import jit
import simulation

# Spatial Points
N = 32
NX = N
NY = N
NZ = 512
TIME_PTS = 140

gridDim = np.array([NX, NY, NZ, TIME_PTS])

# Trap length
L = 12.0
LX = L
LY = L
LZ = 375.0
TIME = 2.500

regionDim = np.array([LX, LY, LZ, TIME])

# Start and end points in space
xa = -LX/2
xb =  xa + LX
ya = -LY/2
yb =  ya + LY
za = - LZ/2
zb = za + LZ

startEnd = np.array([xa, xb, ya, yb, za, zb])

WX = 0. # trapping frequency
WY = 0.0
WZ = 0.

trapVars = np.array([WX, WY, WZ])

OMEGA = 0.0
EPS = 0.0
MOD_TIME = 0.0
G0 = 10.

gVars = np.array([OMEGA, EPS, MOD_TIME, G0])

GPU = False

homogeneousTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

# ---------------- Initial Conditions ----------------

if GPU:
    xp = cp
else:
    xp = np

# Choose homogeneous initial condition
A = .5
psi_init = xp.ones((NX,NY,NZ))*A
#psi_init = cp.asarray(psi_init)

psiPlot, solution = homogeneousTest.realTimeProp(psi_init, True)

# Find max error for validation
analyticAns = A*xp.exp(-1j*G0*np.abs(A)**2*TIME)
maxErr = np.max(np.abs(cp.asnumpy(solution) - cp.asnumpy(analyticAns)))
print(f'Max Error = {maxErr}')
print(f'Norm Loss Ratio = {(homogeneousTest.getNorm( psiPlot[-1,:]) - homogeneousTest.getNorm( psiPlot[0,:]))/homogeneousTest.getNorm( psiPlot[0,:])}')

