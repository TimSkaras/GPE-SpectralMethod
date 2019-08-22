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
        
    5b) Spiffy up documentation
    
    6) Redo simulations for emergence of pereturbations
    
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
        Finds norm for a given state of psi squared along the z-axis
        """
        xp = cp.get_array_module(psiSquared)
        return xp.sqrt(xp.sum(psiSquared*self.hz))
    
    def getEnergy(self, waveFunction, t = 0):
        """
        Finds expectation value of Hamiltonian for given wave function
        """
        xp = cp.get_array_module(waveFunction)
        
        # Potential at each grid point
        energy = xp.sum(-xp.conj(waveFunction)*0.5*self.stencil(waveFunction) \
        + xp.conj(waveFunction)*self.potential*waveFunction \
        + 1./2*self.scatLen(t) * xp.abs(waveFunction)**4)*self.hx*self.hy*self.hz
                    
        return xp.float(xp.abs(energy))
        
    def loadSolution(self, path):
        """
        Retrieves an initial condition or ground state that has been saved from
        the saveSolution method        
        """
        
        # Check to make sure the ground state is compatible with 
        # the simulation object being used to load it
        infoPath = path.split('/')
        targetFile = infoPath[-1]
        infoPath[-1] = 'info.txt'
        infoPath = '/'.join(infoPath)
        
        f = open(infoPath, 'r')
        infoFound = False
        
        for x in f:
            line = x
            line = line.split(' ')
            filename = line[0]
            
            if filename == targetFile:
                data = line[2::2]
                data = [float(q) for q in data]
                
                assert data[0] == self.NX and data[1] == self.NY \
                and data[2] == self.NZ, "Could not load file. Your target file has a number of spatial grid points that differs from your simulation instance"
                
                if not (np.abs(data[3] - self.LX)/self.LX < 10**-8 and np.abs(data[4] - self.LY)/self.LY < 10**-8 \
                and np.abs(data[5] - self.LZ)/self.LZ < 10**-8):
                    print("Warning: Your target file has spatial dimensions that differ from your simulation instance")
                
                if not (np.abs(data[6] - self.WX) < 10**-8 and np.abs(data[7] - self.WY) < 10**-8 \
                and np.abs(data[8] - self.WZ) < 10**-8):
                    print("Warning: Your target file has trapping frequencies that differ from your simulation instance")
                
                if not np.abs(data[9] - self.G0) < 10**-8:
                    print("Warning: Your target file has a scattering length that differs your simulation instance")
                
                break
        else:
            print('Could not find information for target file. Please \
                  ensure you have correctly specified the file you are \
                  trying to load.')
            
                
        psi_init = np.loadtxt(path, dtype=float)
        psi_init = np.reshape(psi_init, (self.NX, self.NY, self.NZ))
        
        return psi_init
    
    # --------------------- Real Time Propagation -------------------------
    
    def reduceIntegrate(self, waveFunction):
        """
        This function takes a three dimensional array of values and redcues it to a
        single dimensional array by integrating over the x and y dimensions
        """
        xp = cp.get_array_module(waveFunction)        
        return xp.sum(xp.abs(waveFunction)**2, axis=(0,1))*self.hx*self.hy
    
    def realTimeProp(self, solution, outputFinalAns = False, outputEnergyPlot = False):
        """
        This function numerically solves the GPE in real time using the spectral method
        
        RETURNS:
            psiPlot -- numpy array with linear z-density as func of time
            outputFinalAns -- by default False. Set to True if you want the method to output the final solution
        
        """
        xp = cp.get_array_module(solution)
        
        psiPlot = np.array([cp.asnumpy(self.reduceIntegrate(solution))]) # Index Convention: psiPlot[time idx][space idx]
        mux = 2*xp.pi/self.LX * xp.arange(-self.NX/2, self.NX/2)
        muy = 2*xp.pi/self.LY * xp.arange(-self.NY/2, self.NY/2)
        muz = 2*xp.pi/self.LZ * xp.arange(-self.NZ/2, self.NZ/2)
        muExpTerm = xp.einsum('i,j,k->ijk', xp.exp(-1j*self.dt*mux**2/2.), xp.exp(-1j*self.dt*muy**2/2.), xp.exp(-1j*self.dt*muz**2/2.))
        muExpTerm = xp.fft.ifftshift(muExpTerm)
        
        energyPlot = np.array([self.getEnergy(solution)])
            
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
            
            # Calculate energy if necessary
            if outputEnergyPlot:
                energyPlot = np.append(energyPlot, self.getEnergy(solution, t = p*self.dt))
            
            # Save Solution for plotting
            psiPlot = np.vstack((psiPlot, cp.asnumpy(self.reduceIntegrate(solution)))) 
        
        output = [psiPlot]
        if outputFinalAns:
            output.append(cp.asnumpy(solution))
        if outputEnergyPlot:
            output.append(energyPlot)
        
        return output
    
    
    # ------------------- Imaginary Time Propagation -------------------------
    
    def normalize(self, waveFunction):
        """
        Takes unnormalized wave function and normalizes it to one
        """
        xp = cp.get_array_module(waveFunction)
        dV = self.hx*self.hy*self.hz
        return waveFunction/xp.sqrt(xp.sum(xp.abs(waveFunction)**2)*dV)
    
#    @jit(nopython=True)
    def stencil(self, solution):
        """
        Finds Laplacian of solution assuming periodic boundary conditions
        
        """
        
        stencilArray = (solution[self.x_forward,:,:] - 2*solution + solution[self.x_back,:,:])/self.hx**2
        stencilArray += (solution[:,self.y_forward,:] - 2*solution + solution[:,self.y_back,:])/self.hy**2
        stencilArray += (solution[:,:,self.z_forward] - 2*solution + solution[:,:,self.z_back])/self.hz**2
    
        return stencilArray
    
    def imagTimeProp(self, solution, tol, max_iter=50000):
        """
        
        Uses a simple forward Euler method to find ground state to within desired tolerance
        
        """
        xp = cp.get_array_module(solution)
        res = 1.0
        iterations = 0
        energyPlot = np.array([self.getEnergy(self.normalize(solution))])

        while res > tol and iterations < max_iter :
            iterations += 1
            new_solution = solution + self.dt*(0.5*self.stencil(solution) - self.potential*solution - self.G0  * np.abs(solution)**2 * solution)
                    
            if iterations % 10 == 0:
                new_solution = self.normalize(new_solution)
                energyPlot = np.append(energyPlot, self.getEnergy(new_solution))
                res = np.abs(energyPlot[-1] - energyPlot[-2])/energyPlot[-2]/self.dt
                
            solution = xp.copy(new_solution)
            
            if iterations % 1000 == 0:
                print(f'Residue = {res}')
        
        print(f'Residue = {res}')
        return self.normalize(solution)
    
    def saveSolution(self, solution, savePath):
        """
        Save the output groundstate from imaginary time propagation method
        
        """
        solution= cp.asnumpy(solution)
        # Convert to 2D array because it is hard to store 3D arrays
        solutionSaveFormat = np.reshape(solution, (self.NX*self.NY,self.NZ))
        np.savetxt(savePath, solutionSaveFormat, fmt='%e')
        
        # Now open info.txt to save information on this file
        infoPath = savePath.split('/')
        infoPath[-1] = 'info.txt'
        infoPath = '/'.join(infoPath)
        
        file = open(infoPath, 'a')
        filename = savePath.split('/')[-1]
        file.write(f'{filename} NX {self.NX} NY {self.NY} NZ {self.NZ} LX {self.LX} LY {self.LY} LZ {self.LZ} WX {self.WX} WY {self.WY} WZ {self.WZ} G {self.G0}\n')
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
    
    def animateSol(self, psiPlot, savePath):
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
        time_text = ax.text( 0.65*self.ZB, 0.9 * y_max, '')
        
        # initialization function: plot the background of each frame
        def init():
            line.set_data([], [])
            return line,
        
        # animation function.  This is called sequentially
        def animate(i):
            y = psiPlot[i,:]
            line.set_data(cp.asnumpy(self.z), y)
            time_text.set_text(f'time = {self.TIME/len(psiPlot) * i:3.3f}')
            return line, time_text
        
        # call the animator.  blit=True means only re-draw the parts that have changed.
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=int(len(psiPlot)), interval=60.0, blit=True)
        
        # save the animation as an mp4.  This requires ffmpeg or mencoder to be
        # installed.  The extra_args ensure that the x264 codec is used, so that
        # the video can be embedded in html5.  You may need to adjust this for
        # your system: for more information, see
        # http://matplotlib.sourceforge.net/api/animation_api.html
        #
        # This will save the animation to file animation.mp4 by default
        
        if savePath:
            anim.save(savePath, fps=120, dpi=150)
    
    # -------------------------- Mode Analysis --------------------------------

    def tkmax(self, psiPlot):
        """
        This function will take psiPlot output from realTimePropagation and then
        find the mode that has the max amplitude and the time at which it reaches
        that maximum
        
        OUTPUT:
            tmax -- time for dominant mode to reach maximum
            kmax -- wavenumber that attains maximum
        """
         
        start_idx = 20
        kPlot = np.fft.fft(psiPlot)
        kPlot = kPlot[:, start_idx:self.NZ//2]
        k = np.array([np.float(j) for j in 2*np.pi*self.NZ/self.LZ* np.arange(start_idx,self.NZ//2)/self.NZ])
        t = np.arange(self.TIME_PTS+1)*self.dt
        kk,tt = np.meshgrid(k, t)
        
        idxMax = np.unravel_index(np.argmax(kPlot), kPlot.shape)
        kmax = k[idxMax[1]]
        tmax = t[idxMax[0]]
        
        return kmax, tmax
    
    
    
    
    
    
    
    
    
    
        