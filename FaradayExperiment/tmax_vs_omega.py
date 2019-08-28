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
TIME = 350

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

omega_list = [1.8, 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2]
kmax_list = np.array([])
tmax_list = np.array([])

for omega in [2.0]:

    OMEGA = omega
    EPS = 0.2
    MOD_TIME = 5*np.pi
    G0 = 1070.
    
    gVars = np.array([OMEGA, EPS, MOD_TIME, G0])
        
    faradayTest = simulation.Simulation(gridDim, regionDim, startEnd, trapVars, gVars, GPU)

    [psiPlot, solution, energyPlot] = faradayTest.realTimeProp(psi_init, True, True)


    kmax, tmax = faradayTest.tkmax(psiPlot)
    kmax_list = np.append(kmax_list, kmax)
    tmax_list = np.append(tmax_list, tmax)
    start_idx = 20


#fig = plt.figure()
#ax = plt.axes(xlim=(1.8,2.2), ylim=(0, 1.5))
#ax.plot(omega_list, kmax_list)
#ax.plot(omega_list, 1.1*np.ones(len(omega_list)))
#ax.set_xlabel(r'$\omega/\omega_\rho$')
#ax.set_ylabel(r'$k_{max}$')
#ax.set_title('Dominant Mode vs. Scattering Length Modulation Frequency')
#ax.legend(['Numerical', 'Analytic'])
#ax.grid()
#
#fig, ax = plt.subplots()
#ax.plot(omega_list, tmax_list)
#ax.set_xlabel(r'$\omega/\omega_\rho$')
#ax.set_ylabel(r'$t_{max}$')
#ax.set_title('Time to Max Amplitude vs. Scattering Length Modulation Frequency')
#ax.grid()
#
#fig = plt.figure()
#ax = plt.axes(xlim=(1.8,2.2), ylim=(0, 1.))
#ax.plot(omega_list, kmax_list /1.74)
#ax.plot(omega_list, 1.1*np.ones(len(omega_list))/1.74)
#ax.set_xlabel(r'$\omega/\omega_\rho$')
#ax.set_ylabel(r'$k_{max}$ $(\mu$m$^{-1})$')
#ax.set_title('Dominant Mode vs. Scattering Length Modulation Frequency')
#ax.legend(['Numerical', 'Analytic'])
#ax.grid()
#
#fig, ax = plt.subplots()
#ax.plot(omega_list, tmax_list * 0.334)
#ax.set_xlabel(r'$\omega/\omega_\rho$')
#ax.set_ylabel(r'$t_{max}$ (ms)')
#ax.set_title('Time to Max Amplitude vs. Scattering Length Modulation Frequency')
#ax.grid()

#dV = faradayTest.hx*faradayTest.hy*faradayTest.hz
#normInit = np.sum(np.abs(psi_init)**2)*dV
#normFinal = np.sum(np.abs(solution)**2)*dV 
#print(f'Norm Error = {(normFinal - normInit)/normInit}')

#faradayTest.surfPlot(psiPlot)

savePath = '../Animations/animation.mp4'

faradayTest.animateSol(psiPlot[::3,:], savePath)

fig, ax = plt.subplots()
t = np.arange(len(energyPlot)) * faradayTest.dt
ax.plot(t, energyPlot)
ax.set_title('Energy vs. Time')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')

#alpha = 10
#kPlot = np.fft.fft(psiPlot)
#kPlot = kPlot[::alpha, start_idx:NZ//4]
#k = np.array([np.float(j) for j in 2*np.pi*NZ/LZ* np.arange(start_idx,NZ//4)/NZ])
#t = np.arange(TIME_PTS+1)*faradayTest.dt
#kk,tt = np.meshgrid(k, t[::alpha])
#fig = plt.figure(2)   # Clear figure 2 window and bring forward
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(kk, tt, np.abs(kPlot), rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
#ax.set_xlabel('Wavenumber')
#ax.set_ylabel('Time')
#ax.set_zlabel('Amplitude')

