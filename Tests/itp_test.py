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
TIME_PTS = 125

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

GPU = True

simulationTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

# ----------- Initial Conditions and Finding G.S. ------------------
if GPU:
    xp = cp
else:
    xp = np
    
sigma = 1.0
sigmaz = xp.sqrt(1/WZ)
psi_init = 1/xp.sqrt(2*xp.pi) * xp.einsum('i,j,k->ijk', xp.exp(-simulationTest.x**2/(2*sigma**2)), \
                     xp.exp(-simulationTest.y**2/(2*sigma**2)), xp.exp(-simulationTest.z**2/(2*sigmaz**2))/sigmaz)
tol = 10**-6

begin = timeit.default_timer()
#solution = simulationTest.imagTimeProp(psi_init, tol)
end = timeit.default_timer()

print(f'Time Elapsed = {end-begin}')

savePath = '../GroundStateSave/test.txt'

#simulationTest.saveSolution(solution, savePath)

loadPath = '../GroundStateSave/gs1.txt'
solution = simulationTest.loadSolution(loadPath)

# Arrays needed for plotting
y_np = simulationTest.hy*np.arange(NY) + ya
z_np = simulationTest.hz*np.arange(NZ) + za
zz,yy = np.meshgrid(z_np, y_np)

fig = plt.figure(2)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
surf = ax.plot_surface(zz, yy, np.sum(np.abs(cp.asnumpy(solution))**2, axis=(0))*simulationTest.hx, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
ax.set_xlabel('Z-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Amplitude)')

