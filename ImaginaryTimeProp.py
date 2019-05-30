
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

test0 = np.array([[[1.2, 1.3, 3.1],
                 [2.1, 2.2, 2.3]],
                 [[1.12, 1.22, 1.32],
                 [2.12, 2.22, 2.32]]])
test = np.reshape(test0, (4,3))
np.savetxt('test.txt', test, fmt='%e')
b = np.loadtxt('test.txt', dtype=float)
b = np.reshape(b, (2,2,3))

def getNorm(waveFunction):
    """
    Finds norm for a given state of psi
    """

    return np.sqrt(np.sum(waveFunction*hz))

def normalize(waveFunction):
    """
    Takes unnormalized wave function and normalizes it to one
    """
    
    return waveFunction/getNorm(waveFunction)

def stencil(solution):
    """
    Finds Laplacian of solution assuming periodic boundary conditions
    
    """
    
    stencilArray = (solution[x_forward,:,:] - 2*solution + solution[x_backward,:,:])/HX**2
    stencilArray += (solution[:,y_forward,:] - 2*solution + solution[:,y_backward,:])/HY**2
    stencilArray += (solution[:,:,z_forward] - 2*solution + solution[:,:,z_backward])/HZ**2

    return stencilArray
    
def imagTimeProp(solution, tol):
    """
    
    Uses a simple forward Euler method to find ground state to within desired tolerance
    
    """
    
    # Setup for method
    potential = np.einsum('i,j,k->ijk', np.exp(0.5*WX**2*(x - (xa+xb)/2.)**2), np.exp(0.5*WY**2*(y - (ya+yb)/2.)**2), 
                              np.exp(0.5*WZ**2*(z-(za+zb)/2.)**2))
    potential = np.log(potential) 
    
    res = 1.0
    iterations = 0
    while res > tol or iterations > 500000:
        iterations += 1
        new_solution = solution + dt*(0.5*stencil(solution) - potential*solution - G * np.abs(solution)**2 * solution)
        res = np.sum(np.abs(new_solution - solution))
    
    return solution


# ---------------- System Parameters ------------------

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
dt = .01
G = 1070.

# Establish grid points
x = hx*np.arange(NX) + xa
y = hy*np.arange(NY) + ya
z = hz*np.arange(NZ) + za

# Arrays needed to find laplacian
x_idx = np.arange(NX)
x_forward = np.roll(x_idx, 1)
x_back = np.roll(x_idx, -1)

y_idx = np.arange(NY)
y_forward = np.roll(y_idx, 1)
y_back = np.roll(y_idx, -1)

z_idx = np.arange(NZ)
z_forward = np.roll(z_idx, 1)
z_back = np.roll(z_idx, -1)


