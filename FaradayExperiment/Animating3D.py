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
NX = 32
NY = 32
NZ = 512
TIME_PTS = 14000

gridDim = np.array([NX, NY, NZ, TIME_PTS])

# Trap length
L = 15.0
LX = 12.0
LY = 12.0
LZ = 375.0
TIME = 350.

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

if GPU:
    xp = cp
else:
    xp = np
    
faradayTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)
# Load Ground State from File
psi_init = xp.asarray(faradayTest.loadSolution('../GroundStateSave/gs1.txt'))

[psiPlot, FinalAns, psiPlot3D] = faradayTest.realTimeProp(psi_init, outputFinalAns=True, output3D=True)

#v = np.real(1/(2j)*(np.conj(FinalAns)*np.gradient(FinalAns) - FinalAns*np.gradient(np.conj(FinalAns))) )

#faradayTest.surfPlot(psiPlot)

savePath = '../Animations/ModulatingPotential1D.mp4'

#faradayTest.animateSol(psiPlot[::3,:], savePath)

#fig = plt.figure()
#ax = fig.add_subplot(111)
#z = psiPlot3D[:,:,0]
##line = ax.plot_wireframe(xx, yy, z,color= 'b', rcount=30, ccount=40)
#line = ax.imshow(psiPlot3D[:,:,0], cmap='hot', interpolation='nearest', extent=[za, zb, xa,xb], aspect=2)

def data(i, z, line):
    z = psiPlot3D[:,:,i]
    ax.clear()
#    ax.set_ylim3d(xa, xb)
#    ax.set_xlim3d(za,zb)
#    ax.set_zlim3d(0,.005)
    line = ax.imshow(z, cmap='hot', interpolation='nearest', extent=[za, zb, xa,xb], aspect=2)
    ax.set_title(f'time = {TIME/(TIME_PTS+1 if TIME_PTS <=1000 else 1001) * i:3.3f}')
    return line,

#ani = animation.FuncAnimation(fig, data, fargs=(z, line), interval=90, blit=False, frames= (TIME_PTS+1 if TIME_PTS <=1000 else 1001))
#ani.save('../Animations/ModulatingPotentialHeatMap.mp4', fps=25, dpi=200)
    
#fig = plt.figure()
#ax = fig.gca(projection='3d')
    
#ax.quiver(X[::4,::4,::32], Z[::4,::4,::32], Y[::4,::4,::32], 
#          v[0,::4,::4,::32], v[2,::4,::4,::32], v[1,::4,::4,::32], 
#          length=0.4, normalize=True)

plt.show()
