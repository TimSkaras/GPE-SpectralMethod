import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import timeit
import numba
from numba import jit


"""
TODO:
    1) Add assert statements to ensure that the input arguments for methods
    are on the correct device (e.g., are on gpu if GPU == True and on CPU
    otherwise)
    
    2) Reduction coefficient
"""

class Simulation:
    
    # --------------------- Set up methods/Auxiliary -------------------------
    
    def __init__(self, gridDim, regionDim, startEnd, trapVars, gVars, GPU):
        """
        This method initializes an instance of the Simulation object
        
        gridDim -- numpy array with dimensions of the finite grid -> [NX, NY, NZ, TIME_PTS]
            NX -- number of grid points along x dimension
            NY -- number of grid points along y dimension
            NZ -- number of grid points along z dimension
            TIME_PTS -- number of points in time (excluding initial point at t=0)
            
        regionDim -- numpy array with size of the region for simulating -> [LX, LY, LZ, TIME]
            LX -- length of region along x dimension
            LY -- length of region along y dimension
            LZ -- length of region along z dimension
            TIME -- total time simulated
            
        startEnd -- numpy array with start and end point of region along each spatial dimension -> [XA, XB, YA, YB, ZA, ZB]
            XA -- starting point of simulation region on x-axis     
            XB -- ending point of simulation region on x-axis
            YA -- starting point of simulation region on y-axis     
            YB -- ending point of simulation region on y-axis  
            ZA -- starting point of simulation region on z-axis     
            ZB -- ending point of simulation region on z-axis  
               
        trapVars -- numpy array of variables for trap frequency -> [WX, WY, WZ]
            WX -- trap frequency along x-dimension
            WY -- trap frequency along y-dimension
            WZ -- trap frequency along z-dimension
            
        gVars -- numpy array for parameters of g (scattering length) -> [OMEGA, EPS, MOD_TIME, G0]
            
            G(t) =  G0*(1 + EPS*sin(OMEGA*t)) if t < MOD_TIME else G0
            
            OMEGA -- angular frequency of time-dep scattering length G(t)
            EPS -- oscillation amplitude of G(t)
            MOD_TIME -- total time for which G(t) should oscillate
            G0 -- magnitude of scattering length
            
        GPU -- boolean value - set to True if you plan to do computation on GPU
                otherwise set to False and computations will be done on CPU
        """
        
        [self.NX, self.NY, self.NZ, self.TIME_PTS] = gridDim
        [self.LX, self.LY, self.LZ, self.TIME] = regionDim
        [self.XA, self.XB, self.YA, self.YB, self.ZA, self.ZB] = startEnd
        [self.WX, self.WY, self.WZ] = trapVars
        [self.OMEGA, self.EPS, self.MOD_TIME, self.G0] = gVars
        
        self.hx = self.LX/self.NX
        self.hy = self.LY/self.NY
        self.hz = self.LZ/self.NZ
        self.dt = self.TIME/self.TIME_PTS
        
        if GPU:
            xp = cp
        else:
            xp = np
            
        # Establish grid points
        self.x = self.hx*xp.arange(self.NX) + self.XA
        self.y = self.hy*xp.arange(self.NY) + self.YA
        self.z = self.hz*xp.arange(self.NZ) + self.ZA
            
        potential = xp.einsum('i,j,k->ijk', xp.exp(0.5*self.WX**2*(self.x - (self.XA+self.XB)/2.)**2), 
                              xp.exp(0.5*self.WY**2*(self.y - (self.YA+self.YB)/2.)**2), 
                              xp.exp(0.5*self.WZ**2*(self.z - (self.ZA+self.ZB)/2.)**2))
        self.potential = xp.log(potential) 
        
        # Arrays needed to find laplacian
        x_idx = xp.arange(self.NX)
        self.x_forward = xp.roll(x_idx, 1)
        self.x_back = xp.roll(x_idx, -1)
        
        y_idx = xp.arange(self.NY)
        self.y_forward = xp.roll(y_idx, 1)
        self.y_back = xp.roll(y_idx, -1)
        
        z_idx = xp.arange(self.NZ)
        self.z_forward = xp.roll(z_idx, 1)
        self.z_back = xp.roll(z_idx, -1)
        
    def scatLen(self, t):
        """
        Returns time-dependent scattering length at a given time t
        """
        return  (self.G0* (1.0 + self.EPS*np.sin(self.OMEGA*t)) if t < self.MOD_TIME else self.G0)
    
    def getNorm(self, psiSquared):
        """
        Finds norm for a given state of psi squared
        """
        xp = cp.get_array_module(psiSquared)
        return xp.sqrt(xp.sum(psiSquared*self.hz))
    
    def getEnergy(self, waveFunction):
        """
        Finds expectation value of Hamiltonian for given wave function
        """
        xp = cp.get_array_module(waveFunction)
        
        # Potential at each grid point
        energy = xp.sum(-xp.conj(waveFunction)*0.5*stencil(waveFunction) 
        + xp.conj(waveFunction)*self.potential*waveFunction 
        + self.G0 * xp.abs(waveFunction)**4)*self.hx*self.hy*self.hz
    
        return energy
    
    def loadSolution(self, path):
        """
        Retrieves an initial condition or ground state that has been saved from
        the saveSolution method        
        """
        pass
    
    # --------------------- Real Time Propagation -------------------------
    
    def reduceIntegrate(self, waveFunction):
        """
        This function takes a three dimensional array of values and redcues it to a
        single dimensional array by integrating over the x and y dimensions
        """
        xp = cp.get_array_module(waveFunction)        
        return xp.sum(xp.abs(waveFunction)**2, axis=(0,1))*self.hx*self.hy
    
    def realTimeProp(self, solution):
        """
        This function numerically solves the GPE in real time using the spectral method
        
        RETURNS:
            psiPlot -- numpy array with linear z-density as func of time
        
        """
        xp = cp.get_array_module(solution)
        
        psiPlot = np.array([cp.asnumpy(self.reduceIntegrate(solution))]) # Index Convention: psiPlot[time idx][space idx]
        mux = 2*xp.pi/self.LX * xp.arange(-self.NX/2, self.NX/2)
        muy = 2*xp.pi/self.LY * xp.arange(-self.NY/2, self.NY/2)
        muz = 2*xp.pi/self.LZ * xp.arange(-self.NZ/2, self.NZ/2)
        muExpTerm = xp.einsum('i,j,k->ijk', xp.exp(-1j*self.dt*mux**2/2.), xp.exp(-1j*self.dt*muy**2/2.), xp.exp(-1j*self.dt*muz**2/2.))
        muExpTerm = xp.fft.ifftshift(muExpTerm)
            
        
        for p in range(self.TIME_PTS):
            
            # Step One -- potential and interaction term
            expTerm = xp.exp(-1j* (self.potential + self.scatLen(p*self.dt)*xp.abs(solution)**2)*self.dt/2.0)
            solution = expTerm*solution
                        
            # Step Two -- kinetic term
            fourierCoeffs = xp.fft.fftn(solution)
            fourierCoeffs = fourierCoeffs*muExpTerm
            solution = xp.fft.ifftn(fourierCoeffs)
            
            # Step Three -- potential and interaction term again
            expTerm = xp.exp(-1j* (self.potential + self.scatLen((p + 0.5)*self.dt)*xp.abs(solution)**2)*self.dt/2.0)
            solution = expTerm*solution
            
            # Save Solution for plotting
            psiPlot = np.vstack((psiPlot, cp.asnumpy(self.reduceIntegrate(solution)))) 
            
        return psiPlot
    
    
    # ------------------- Imaginary Time Propagation -------------------------
    
    def normalize(self, waveFunction):
        """
        Takes unnormalized wave function and normalizes it to one
        """
        xp = cp.get_array_module(waveFunction)
        return waveFunction/self.getNorm(xp.abs(waveFunction)**2)
    
    @jit(nopython=True)
    def stencil(self, solution):
        """
        Finds Laplacian of solution assuming periodic boundary conditions
        
        """
        
        stencilArray = (solution[self.x_forward,:,:] - 2*solution + solution[self.x_back,:,:])/self.hx**2
        stencilArray += (solution[:,self.y_forward,:] - 2*solution + solution[:,self.y_back,:])/self.hy**2
        stencilArray += (solution[:,:,self.z_forward] - 2*solution + solution[:,:,self.z_back])/self.hz**2
    
        return stencilArray
    
    def imagTimeProp(self, solution, tol):
        """
        
        Uses a simple forward Euler method to find ground state to within desired tolerance
        
        """
        xp = cp.get_array_module(solution)
        res = 1.0
        iterations = 0
        energyPlot = np.array([getEnergy(normalize(solution))])
        max_iter = 50000

        while res > tol and iterations < max_iter :
            iterations += 1
            new_solution = solution + self.dt*(0.5*self.stencil(solution) - self.potential*solution - self.G0 * np.abs(solution)**2 * solution)
                    
            if iterations % 10 == 0:
                new_solution = self.normalize(new_solution)
                energyPlot = np.append(energyPlot, self.getEnergy(new_solution))
                res = np.abs(energyPlot[-1] - energyPlot[-2])/energyPlot[-2]/dt
                
            solution = xp.copy(new_solution)
            
            if iterations % 1000 == 0:
                print(f'Residue = {res}')
        
        print(f'Residue = {res}')
        return self.normalize(solution)
    
    def saveSolution(self, solution, savePath):
        """
        Save the output groundstate from imaginary time propagation method
        
        """
        xp = cp.get_array_module(solution)
        # Convert to 2D array because it is hard to store 3D arrays
        solutionSaveFormat = np.reshape(solution, (self.NX*self.NY,self.NZ))
        np.savetxt(savePath, solutionSaveFormat, fmt='%e')
        file = open(savePath, 'a')
        file.write(f'{savePath} NX {NX} NY {NY} NZ {NZ} LX {LX} LY {LY} LZ {LZ} WX {WX} WY {WY} WZ {WZ}\n')
        file.close()
        
    
    # -------------- Plotting, Animation, & Visualization --------------------
    
    def surfPlot(self, psiPlot):
        """
        Plots the linear z-density of the solution that resulted from realTimeProp
        
        INTPUTS:
            psiPlot -- 2D numpy array of the integrated z density from realTimeProp
        
        """
        xx,tt = np.meshgrid(cp.asnumpy(self.z),np.arange(self.TIME_PTS+1)*self.dt)

        fig = plt.figure(2)   # Clear figure 2 window and bring forward
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(xx, tt, psiPlot, rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
        ax.set_xlabel('Position')
        ax.set_ylabel('Time')
        ax.set_zlabel('Amplitude')
    
    def animateSol(self, psiPlot, savePath=''):
        """
        This function will generate an animation of the time evolution of psi
        as given in psiPlot. This animation can be saved by passing in the 
        desired path for animation to be saved to.
        
        If you don't want to save it, don't pass in the savePath variable
        """
        
        y_min = 0
        y_max = np.max(psiPlot)
        
        fig = plt.figure(figsize=(8, 6), dpi=80)
        fig.set_size_inches(8, 6, forward=True)
        ax = plt.axes(xlim=(self.ZA, self.ZB), ylim=(y_min, y_max))
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
            line.set_data(cp.asnumpy(self.z), y)
            return line,
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=int(self.TIME_PTS), interval=60.0, blit=True)
        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        #
        # This will save the animation to file animation.mp4 by default
        
        if savePath:
            anim.save(savePath, fps=150, dpi=150)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        