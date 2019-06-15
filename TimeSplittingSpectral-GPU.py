"""
This code solves the equation
i du/dt = -1/2 d^2u/dx^2 + 1/2*w**2*x**2*u + g|u|^2 u

this code is three dimensional and has periodic boundary conditions

The two end points for the solution a and b are at a=0 and b=L

TODO:
    1) Although the scattering length does change in time it might not properly
    account for the fact that the time is not the same at different steps 
    within the Time Splitting Spectral Method
    2) Display simulated time on each frame of animation

"""

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import timeit

# ---------------- Real Time Propagation via Spectral Method

def getNorm(psiSquared):
    """
    Finds norm for a given state of psi squared
    """

    return np.sqrt(np.sum(psiSquared*hz))

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
        expTerm = cp.exp(-1j* (potential + GFunc(p*dt)*cp.abs(solution)**2)*dt/2.0)
        solution = expTerm*solution
                    
                    
        # Step Two -- kinetic term
        fourierCoeffs = cp.fft.fftn(solution)
        fourierCoeffs = fourierCoeffs*muExpTerm
        solution = cp.fft.ifftn(fourierCoeffs)
        
        # Step Three -- potential and interaction term again
        expTerm = cp.exp(-1j* (potential + GFunc((p + 0.5)*dt)*cp.abs(solution)**2)*dt/2.0)
        solution = expTerm*solution
        
        # Save Solution for plotting
#        np_solution = cp.asnumpy(solution)
        psiPlot = np.vstack((psiPlot, cp.asnumpy(reduceIntegrateCUDA(solution)))) 
        
    return psiPlot
    
    

# ----------- Important variables --------------

surf_plot_on = 0

# Animation Input Parameters
animation_on = 0
animation_filename = 'Animations/animation.mp4' # filename for animation output

# Spatial Points
N = 32
NX = N
NY = N
NZ = 512

# Trap length
L = 12.0
LX = L
LY = L
LZ = 375.0

TIME = 250.0

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

TIME_PTS = 14000
RED_COEFF = 1

dt = TIME/TIME_PTS
OMEGA = 2.0
EPS = 0.2
MOD_TIME = 5*np.pi
G = 1070.
GFunc = lambda t: (G* (1.0 + EPS*np.sin(OMEGA*t)) if t < MOD_TIME else G) # interaction strength parameter

W = 1.0
WX = 1. # trapping frequency
WY = 1.
WZ = 7./476.

# Establish grid points
x = hx*cp.arange(NX) + xa
y = hy*cp.arange(NY) + ya
z = hz*cp.arange(NZ) + za

y_np = hy*np.arange(NY) + ya
z_np = hz*np.arange(NZ) + za

# ---------------- Initial Conditions ----------------

# Example Initial Condition
# Gaussian initial
sigma = 1.0
sigmaz = np.sqrt(1/WZ)
psi_init = 1/cp.sqrt(2*cp.pi) * cp.einsum('i,j,k->ijk', cp.exp(-x**2/(2*sigma**2)), cp.exp(-y**2/(2*sigma**2)), cp.exp(-z**2/(2*sigmaz**2)))

# Load Ground State from File
psi_init = np.loadtxt('GroundStateSave/gs1.txt', dtype=float)
psi_init = np.reshape(psi_init, (NX, NY, NZ))

## Plot the z-axis of the ground state
#fig, ax = plt.subplots()
#ax.plot(z_np, np.sum(np.abs(psi_init)**2, axis=(0,1)) *hx*hy*hz)

# ------------------ Solver ----------------------

# Move initial condition to GPU and run simulation
psi_init = cp.asarray(psi_init)
begin = timeit.default_timer()
psiPlot = realTimePropagation(psi_init)
end = timeit.default_timer()
        
print(f'Time Elapsed = {end-begin}')
print(f'Grid Points = {NX*NY*NZ}')

## Norm conservation as function of time
norm_time = [getNorm(psiPlot[j, :]) for j in range(len(psiPlot))]
print(f'Norm Ratio = {norm_time[-1]/norm_time[0] - 1}\n')

print(f'Estimated Runtime = {(end-begin)/TIME_PTS * 14000 / 60} min')

# ----------------- Display Results --------------

#homExact = A*cp.exp(-1j*G*cp.abs(A)**2 * TIME)
#print('Homogeneous Solution = ' + f'{homExact}')

#fig = plt.figure(1)   # Clear figure 2 window and bring forward
#plt.plot(np.arange(TIME_PTS)*dt, norm_time)
#plt.title("Norm vs Time")
#plt.ylabel("Norm of Wave Function")
#plt.xlabel('Time')

xx,tt = np.meshgrid(z_np,np.arange(TIME_PTS+1)*dt)
if surf_plot_on:
    fig = plt.figure(2)   # Clear figure 2 window and bring forward
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xx, tt, psiPlot, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
    ax.set_xlabel('Position')
    ax.set_ylabel('Time')
    ax.set_zlabel('Amplitude)')
        
        
# ----------- Animation -------------
# This code is for generating an animation of psiPlot
#
# This code will save the animation to an mp4
# First set up the figure, the axis, and the plot element we want to animate

if animation_on:
    y_min = 0
    y_max = np.max(psiPlot)
    
    fig = plt.figure(num=3, figsize=(8, 6), dpi=80)
    fig.set_size_inches(8, 6, forward=True)
    ax = plt.axes(xlim=(za, zb), ylim=(y_min, y_max))
    ax.set_title('Time Evolution of Linear Z-Density')
    ax.set_xlabel('z')
    ax.set_ylabel(r'$|\psi|^2$')
    line, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return line,
    
    # animation function.  This is called sequentially
    def animate(i):
        y = psiPlot[i,:]
        line.set_data(z_np, y)
        return line,
    
    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=int(TIME_PTS/RED_COEFF), interval=60.0, blit=True)
    
    # save the animation as an mp4.  This requires ffmpeg or mencoder to be
    # installed.  The extra_args ensure that the x264 codec is used, so that
    # the video can be embedded in html5.  You may need to adjust this for
    # your system: for more information, see
    # http://matplotlib.sourceforge.net/api/animation_api.html
    #
    # This will save the animation to file animation.mp4 by default
    anim.save(animation_filename, fps=150, dpi=150)


# -------------- Mode Analysis -------------------
    
k = 2*np.pi*NZ/LZ * np.arange(0, NZ//2)/NZ
kk,tt = np.meshgrid(k,np.arange(0,TIME_PTS+1, 100)*dt)
kspace = np.abs(np.fft.fft(psiPlot))[::100, 0:NZ//2]

fig = plt.figure(4)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
surf = ax.plot_surface(kk, tt, kspace, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Time')
ax.set_zlabel('Amplitude)')