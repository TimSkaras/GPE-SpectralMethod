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
N = 64
NX = N
NY = N
NZ = N
TIME_PTS = 1400

gridDim = np.array([NX, NY, NZ, TIME_PTS])

# Trap length
L = 15.0
LX = L
LY = L
LZ = L
TIME = 2.5*10

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
WZ = 1.

trapVars = np.array([WX, WY, WZ])

OMEGA = 2.0
EPS = 0.0
MOD_TIME = 5*np.pi
G0 = 1070.

gVars = np.array([OMEGA, EPS, MOD_TIME, G0])

GPU = True

faradayTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

# ---------------- Initial Conditions ----------------
if GPU:
    xp = cp
else:
    xp = np


# Load Ground State from File
psi_init = xp.asarray(faradayTest.loadSolution('../GroundStateSave/gs2.txt'))

[psiPlot, solution, energyPlot] = faradayTest.realTimeProp(psi_init, True, True)

dV = faradayTest.hx*faradayTest.hy*faradayTest.hz
normInit = np.sum(np.abs(psi_init)**2)*dV
normFinal = np.sum(np.abs(solution)**2)*dV 
print(f'Norm Error = {(normFinal - normInit)/normInit}')

#faradayTest.surfPlot(psiPlot)

savePath = '../Animations/animation.mp4'

#faradayTest.animateSol(psiPlot, savePath)

#faradayTest.surfPlot(psiPlot)


fig, ax = plt.subplots()
t = np.arange(len(energyPlot)) * faradayTest.dt
ax.plot(t, energyPlot)
ax.set_title('Energy vs. Time')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
