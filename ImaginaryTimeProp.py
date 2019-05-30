
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
    
    stencilArray = (solution[x_forward,:,:] - 2*solution + solution[x_back,:,:])/hx**2
    stencilArray += (solution[:,y_forward,:] - 2*solution + solution[:,y_back,:])/hy**2
    stencilArray += (solution[:,:,z_forward] - 2*solution + solution[:,:,z_back])/hz**2

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
    while res > tol and iterations < max_iter :
        iterations += 1
        new_solution = solution + dt*(0.5*stencil(solution) - potential*solution - G * np.abs(solution)**2 * solution)
        res = np.sum(np.abs(new_solution - solution))
        
        if iterations % 20 == 0:
            new_solution = normalize(new_solution)
            
        solution = np.copy(new_solution)
    
    print(f'Tolerance = {res}')
    return normalize(solution)


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
dt = .05
G = 1070.

#Trap Configuration
W = 1.0
WX = 1. # trapping frequency
WY = 1.
WZ = 7./476.

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

# ----------- Initial Conditions and Finding G.S. ------------------

sigma = 1.0
sigmaz = np.sqrt(1/WZ)
psi_init = 1/np.sqrt(2*np.pi) * np.einsum('i,j,k->ijk', np.exp(-x**2/(2*sigma**2)), np.exp(-y**2/(2*sigma**2)), np.exp(-z**2/(2*sigmaz**2)))
tol = 10**-2
max_iter = 50000

solution = imagTimeProp(psi_init, tol)


# ----------------- Plotting Results --------------------------

# Arrays needed for plotting
y_np = hy*np.arange(NY) + ya
z_np = hz*np.arange(NZ) + za
zz,yy = np.meshgrid(z_np, y_np)

fig = plt.figure(2)   # Clear figure 2 window and bring forward
ax = fig.gca(projection='3d')
surf = ax.plot_surface(zz, yy, np.sum(np.abs(solution)**2, axis=(0))*hx, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
ax.set_xlabel('Z-Axis')
ax.set_ylabel('Y-Axis')
ax.set_zlabel('Amplitude)')