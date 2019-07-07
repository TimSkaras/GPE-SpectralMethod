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

planeWaveTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

# ---------------- Initial Conditions ----------------

if GPU:
    xp = cp
else:
    xp = np

# Choose homogeneous initial condition
A = .01
kx = 0.0
ky = 2*np.pi/LY*3
kz = 2*np.pi/LZ*2
k = [kx, ky, kz]
psi_init = A * xp.einsum('i,j,k->ijk',xp.exp(1j*kx*planeWaveTest.x), 
                     xp.exp(1j*ky*planeWaveTest.y), 
                     xp.exp(1j*kz*planeWaveTest.z))
#psi_init = cp.asarray(psi_init)

psiPlot, solution = planeWaveTest.realTimeProp(psi_init, True)

# Find max error for validation
analyticAns = psi_init * np.exp(-1j*(np.linalg.norm(k)**2/2. + G0*np.abs(A)**2)*TIME)
maxErr = xp.max(xp.abs(solution - analyticAns))
print(f'Max Error = {maxErr}')

