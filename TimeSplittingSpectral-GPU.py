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

def reduceIntegrateCUDA(waveFunction):
    """
    This function takes a three dimensional array of values and redcues it to a
    single dimensional array by integrating over the x and y dimensions
    """
    
    return cp.sum(cp.abs(waveFunction)**2, axis=(0,1))*hx*hy

def reduceIntegrate(waveFunction):
    """
    This function takes a three dimensional array of values and redcues it to a
    single dimensional array by integrating over the x and y dimensions
    """
    
    return np.sum(np.abs(waveFunction)**2, axis=(0,1))*hx*hy
    
def realTimePropagation(solution):
    """
    This function numerically solves the GPE in real time using the spectral method
    
    """
#    np_solution = cp.asnumpy(solution)
    psiPlot = np.array([cp.asnumpy(reduceIntegrateCUDA(solution))]) # Index Convention: psiPlot[time idx][space idx]
    mux = 2*cp.pi/LX * cp.arange(-NX/2, NX/2)
    muy = 2*cp.pi/LY * cp.arange(-NY/2, NY/2)
    muz = 2*cp.pi/LZ * cp.arange(-NZ/2, NZ/2)
    muExpTerm = cp.einsum('i,j,k->ijk', cp.exp(-1j*dt*mux**2/2.), cp.exp(-1j*dt*muy**2/2.), cp.exp(-1j*dt*muz**2/2.))
    muExpTerm = cp.fft.ifftshift(muExpTerm)
        
    potential = cp.einsum('i,j,k->ijk', cp.exp(0.5*WX**2*(x - (xa+xb)/2.)**2), cp.exp(0.5*WY**2*(y - (ya+yb)/2.)**2), 
                              cp.exp(0.5*WZ**2*(z-(za+zb)/2.)**2))
    potential = cp.log(potential) 
    
    for p in range(TIME_PTS):
        
        # Step One -- potential and interaction term
        expTerm = cp.exp(-1j* (potential + G*cp.abs(solution)**2)*dt/2.0)
        solution = expTerm*solution
                    
                    
        # Step Two -- kinetic term
        fourierCoeffs = cp.fft.fftn(solution)
        fourierCoeffs = fourierCoeffs*muExpTerm
        solution = cp.fft.ifftn(fourierCoeffs)
        
        # Step Three -- potential and interaction term again
        expTerm = cp.exp(-1j* (potential + G*cp.abs(solution)**2)*dt/2.0)
        solution = expTerm*solution
        
        # Save Solution for plotting
#        np_solution = cp.asnumpy(solution)
        psiPlot = np.vstack((psiPlot, cp.asnumpy(reduceIntegrateCUDA(solution)))) 
        
    return psiPlot
    
    

# ----------- Important variables --------------

# Spatial Points
N = 32
NX = N
NY = N
NZ = 512

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

TIME_PTS = 100

dt = TIME/TIME_PTS
G = 10. # interaction strength parameter
W = 1.0
WX = W # trapping frequency
WY = W
WZ = W
A = .30 + .40j # Coefficient for homogeneous solution

# Establish grid points
x = hx*cp.arange(NX) + xa
y = hy*cp.arange(NY) + ya
z = hz*cp.arange(NZ) + za

# ------------- Initial Conditions -------------

## Soliton solution

## Homogeneous solution
#psi_init = A*cp.ones((NX, NY, NZ))

# Gaussian initial
sigma = 1.0
psi_init = 1/cp.sqrt(2*cp.pi) * cp.einsum('i,j,k->ijk', cp.exp(-x**2/(2*sigma**2)), cp.exp(-y**2/(2*sigma**2)), cp.exp(-z**2/(2*sigma**2)))

## Plane Wave
#mu = 2*cp.pi*cp.array([0/LX, 1/LY, -2/LZ])
#psi_init = cp.ones((NX, NY, NZ), dtype=cp.complex_)
#for i in cp.arange(NX):
#    for j in cp.arange(NY):
#        for k in cp.arange(NZ):
#            psi_init[i,j,k] = cp.exp(1j*(mu[0]*x[i] + mu[1]*y[j] + mu[2]*z[k]))
#            

begin = timeit.default_timer()
psiPlot = realTimePropagation(psi_init)
end = timeit.default_timer()
        

print(f'Time Elapsed = {end-begin}')
print(f'Grid Points = {NX*NY*NZ}')

## Norm conservation as function of time
norm_time = [getNorm(psiPlot[j, :]) for j in range(len(psiPlot))]
print(f'Norm Ratio = {norm_time[-1]/norm_time[0] - 1}\n')

print(f'Estimated Runtime = {(end-begin)/TIME_PTS * 32000 / 60} min')

#homExact = A*cp.exp(-1j*G*cp.abs(A)**2 * TIME)
#print('Homogeneous Solution = ' + f'{homExact}')

#fig = plt.figure(1)   # Clear figure 2 window and bring forward
#plt.plot(np.arange(TIME_PTS)*dt, norm_time)
#plt.title("Norm vs Time")
#plt.ylabel("Norm of Wave Function")
#plt.xlabel('Time')

z_np = hz*np.arange(NZ) + za
xx,tt = np.meshgrid(z_np,np.arange(TIME_PTS+1)*dt)
fig = plt.figure(2)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xx, tt, psiPlot, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
ax.set_xlabel('Position')
ax.set_ylabel('Time')
ax.set_zlabel('Amplitude)')