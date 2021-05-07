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
TIME_PTS = 850

gridDim = np.array([NX, NY, NZ, TIME_PTS])

# Trap length
L = 15.0
LX = L
LY = L
LZ = L
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

W = 1.
WX = W # trapping frequency
WY = W
WZ = W

trapVars = np.array([WX, WY, WZ])

OMEGA = 2.0
EPS = 0.0
MOD_TIME = 5*np.pi
G0 = 2000.

gVars = np.array([OMEGA, EPS, MOD_TIME, G0])

GPU = True

simulationTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

# ----------- Initial Conditions and Finding G.S. ------------------
if GPU:
    xp = cp
else:
    xp = np
    
sigma = xp.sqrt(1/W)
psi_init = 1/xp.sqrt(2*xp.pi) * xp.einsum('i,j,k->ijk', xp.exp(-simulationTest.x**2/(2*sigma**2)), \
                     xp.exp(-simulationTest.y**2/(2*sigma**2)), xp.exp(-simulationTest.z**2/(2*sigma**2))/sigma)
tol = 10**-15

begin = timeit.default_timer()
solution = simulationTest.imagTimeProp(psi_init, tol)
end = timeit.default_timer()
dV = simulationTest.hx*simulationTest.hy*simulationTest.hz
print(f'Numerically Calculated Energy = {simulationTest.getEnergy(solution)}')
print(f'Numerical Solution Norm = {np.sum(np.abs(cp.asnumpy(solution))**2) *dV}')

# Calculate analytic with TF ground state solution
mu = 0.5 * (15./4 * G0 * WY * WZ/WX**2/np.pi)**(2./5) 
TFgs = (mu - cp.asnumpy(simulationTest.potential))/G0
idxZero = np.where(TFgs < 0)
TFgs[idxZero] = 0.0
TFgs = np.sqrt(TFgs)
TFgs = TFgs/np.sqrt(np.sum(TFgs**2) *dV)
print(f'Analytic Answer Energy = {simulationTest.getEnergy(xp.asarray(TFgs ))}')
print(f'norm = {np.sum(TFgs**2) *dV}')
print(f'Max Error = {np.max(np.abs(TFgs - cp.asnumpy(solution)))}')

y_np = simulationTest.hy*np.arange(NY) + ya
z_np = simulationTest.hz*np.arange(NZ) + za
zz,yy = np.meshgrid(z_np, y_np)



fig = plt.figure(2)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
ax.plot_wireframe(zz, yy, np.sum(np.abs(cp.asnumpy(solution))**2, axis=(0))*simulationTest.hx)
ax.set_xlabel('Z-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel(r'$|\psi|^2$')
ax.set_title('Numerical Solution')


fig = plt.figure(3)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
#surf = ax.plot_surface(zz, yy, np.sum(np.abs(cp.asnumpy(TFgs))**2, axis=(0))*simulationTest.hx, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
ax.plot_wireframe(zz, yy, np.sum(np.abs(cp.asnumpy(TFgs))**2, axis=(0))*simulationTest.hx)
ax.set_xlabel('Z-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel(r'$|\psi|^2$')
ax.set_title('Analytic Approximation')