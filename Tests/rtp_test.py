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

animation_filename = 'Animations/animation.mp4' # filename for animation output

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

WX = 1. # trapping frequency
WY = 1.
WZ = 7./476.

trapVars = np.array([WX, WY, WZ])

OMEGA = 2.0
EPS = 0.2
MOD_TIME = 5*np.pi
G0 = 1070.

gVars = np.array([OMEGA, EPS, MOD_TIME, G0])

GPU = False

faradayTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

# ---------------- Initial Conditions ----------------

# Load Ground State from File
psi_init = np.loadtxt('../GroundStateSave/gs1.txt', dtype=float)
psi_init = np.reshape(psi_init, (NX, NY, NZ))
#psi_init = cp.asarray(psi_init)

[psiPlot] = faradayTest.realTimeProp(psi_init)

#faradayTest.surfPlot(psiPlot)

savePath = '../Animations/animation.mp4'

faradayTest.animateSol(psiPlot, savePath)

