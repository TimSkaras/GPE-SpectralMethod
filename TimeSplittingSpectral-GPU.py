"""
This code is tests out the time-splitting spectral method in python because it 
is easier to develop code for the first time in python/Matlab

This code solves the equation
i du/dt = -1/2 d^2u/dx^2 + 1/2*w**2*x**2*u + g|u|^2 u

this code is one dimensional and has periodic boundary conditions

The two end points for the solution a and b are at a=0 and b=L

"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numba
from numba import jit
import timeit

def getNorm(waveFunction):
    """
    Finds norm for a given state of psi
    """

    return np.sqrt(np.sum(waveFunction*hz))

def reduceIntegrate(waveFunction):
    """
    This function takes a three dimensional array of values and redcues it to a
    single dimensional array by integrating over the x and y dimensions
    """
    
    return np.sum(np.abs(waveFunction)**2, axis=(0,1))*hx*hy
    
def realTimePropagation(psi_initial):
    """
    This function numerically solves the GPE in real time using the spectral method
    
    """
    
    solution = np.copy(psi_initial)
    psiPlot = np.array([reduceIntegrate(psi_initial)]) # Index Convention: psiPlot[time idx][space idx]
    mux = 2*np.pi/LX * np.arange(-NX/2, NX/2)
    muy = 2*np.pi/LY * np.arange(-NY/2, NY/2)
    muz = 2*np.pi/LZ * np.arange(-NZ/2, NZ/2)
    muExpTerm = np.einsum('i,j,k->ijk', np.exp(-1j*dt*mux**2/2.), np.exp(-1j*dt*muy**2/2.), np.exp(-1j*dt*muz**2/2.))
    muExpTerm = np.fft.ifftshift(muExpTerm)
        
    potential = np.einsum('i,j,k->ijk', np.exp(0.5*WX**2*(x - (xa+xb)/2.)**2), np.exp(0.5*WY**2*(y - (ya+yb)/2.)**2), 
                              np.exp(0.5*WZ**2*(z-(za+zb)/2.)**2))
    potential = np.log(potential) 
    
    fourierCoeffsManual = np.zeros((NX, NY, NZ), dtype=np.complex_)
    invFourier = np.zeros((NX, NY, NZ), dtype=np.complex_)
    
    for p in range(TIME_PTS):
        
        # Step One -- potential and interaction term
        test = np.copy(solution)
        expTerm = np.exp(-1j* (potential + G*np.abs(test)**2)*dt/2.0)
        solution = expTerm*solution
                    
                    
        # Step Two -- kinetic term
        fourierCoeffs = np.fft.fftn(solution)
        fourierCoeffs = fourierCoeffs*muExpTerm
        solution = np.fft.ifftn(fourierCoeffs)
        
        # Step Three -- potential and interaction term again
        expTerm = np.exp(-1j* (potential + G*np.abs(solution)**2)*dt/2.0)
        solution = expTerm*solution
        
        # Save Solution for plotting
        psiPlot = np.vstack((psiPlot, reduceIntegrate(solution))) 
        
    return psiPlot
    
    

# ----------- Important variables --------------

# Spatial Points
N = 80
NX = N
NY = N
NZ = N

# Trap length
L = 8.0
LX = L
LY = L
LZ = L

TIME = 8.0

# Start and end points in space
xa = -LX/2
xb =  xa + LX
ya = -LY/2
yb =  ya + LY
za = - LZ/2
zb = za + LZ

# Spatial Steps
hx = LX/NX
hy = LY/NY
hz = LZ/NZ

TIME_PTS = 50

dt = TIME/TIME_PTS
G = 10. # interaction strength parameter
W = 1.0
WX = W # trapping frequency
WY = W
WZ = W
A = .30 + .40j # Coefficient for homogeneous solution

# Establish grid points
x = hx*np.arange(NX) + xa
y = hy*np.arange(NY) + ya
z = hz*np.arange(NZ) + za

# ------------- Initial Conditions -------------

## Soliton solution

## Homogeneous solution
#psi_init = A*np.ones((NX, NY, NZ))

# Gaussian initial
sigma = 1.0
psi_init = 1/np.sqrt(2*np.pi) * np.einsum('i,j,k->ijk', np.exp(-x**2/(2*sigma**2)), np.exp(-y**2/(2*sigma**2)), np.exp(-z**2/(2*sigma**2)))

## Plane Wave
#mu = 2*np.pi*np.array([0/LX, 1/LY, -2/LZ])
#psi_init = np.ones((NX, NY, NZ), dtype=np.complex_)
#for i in np.arange(NX):
#    for j in np.arange(NY):
#        for k in np.arange(NZ):
#            psi_init[i,j,k] = np.exp(1j*(mu[0]*x[i] + mu[1]*y[j] + mu[2]*z[k]))
#            

begin = timeit.default_timer()
psiPlot = realTimePropagation(psi_init)
end = timeit.default_timer()
        

print(f'Time Elapsed = {end-begin}')
print(f'Grid Points = {NX*NY*NZ}')

## Norm conservation as function of time
norm_time = [getNorm(psiPlot[j, :]) for j in range(len(psiPlot))]
print(f'Norm Ratio = {norm_time[-1]/norm_time[0] - 1}')

#homExact = A*np.exp(-1j*G*np.abs(A)**2 * TIME)
#print('Homogeneous Solution = ' + f'{homExact}')

#fig = plt.figure(1)   # Clear figure 2 window and bring forward
#plt.plot(np.arange(TIME_PTS)*dt, norm_time)
#plt.title("Norm vs Time")
#plt.ylabel("Norm of Wave Function")
#plt.xlabel('Time')


xx,tt = np.meshgrid(z,np.arange(TIME_PTS+1)*dt)
fig = plt.figure(2)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, tt, psiPlot, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
ax.set_xlabel('Position')
ax.set_ylabel('Time')
ax.set_zlabel('Amplitude)')