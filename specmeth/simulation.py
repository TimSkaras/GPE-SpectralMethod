import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import timeit
import numba
from numba import jit

class Simulation:
    
    # --------------------- Set up methods/Auxiliary -------------------------
    
    def __init__(self, gridDim, regionDim, startEnd, trapVars, gVars):
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
        
    def scatLen(self, t):
        """
        Returns time-dependent scattering length at a given time t
        """
        return  (self.G0* (1.0 + self.EPS*np.sin(self.OMEGA*t)) if t < self.MOD_TIME else self.G0)
    
    # --------------------- Real Time Propagation -------------------------
    
    def getNorm(self, psiSquared):
        """
        Finds norm for a given state of psi squared
        """
        xp = cp.get_array_module(psiSquared)
        return xp.sqrt(xp.sum(psiSquared*self.hz))
    
    def reduceIntegrate(self, waveFunction):
        """
        This function takes a three dimensional array of values and redcues it to a
        single dimensional array by integrating over the x and y dimensions
        """
        xp = cp.get_array_module(psiSquared)        
        return xp.sum(xp.abs(waveFunction)**2, axis=(0,1))*self.hx*self.hy
    
    def realTimeProp(self, solution):
        """
        This function numerically solves the GPE in real time using the spectral method
        
        RETURNS:
            psiPlot -- numpy array with linear z-density as func of time
        
        """
        xp = cp.get_array_module(solution)
        
        # Establish grid points
        x = self.hx*xp.arange(self.NX) + self.XA
        y = self.hy*xp.arange(self.NY) + self.YA
        z = self.hz*xp.arange(self.NZ) + self.ZA
        
        psiPlot = np.array([cp.asnumpy(reduceIntegrate(solution))]) # Index Convention: psiPlot[time idx][space idx]
        mux = 2*xp.pi/self.LX * xp.arange(-self.NX/2, self.NX/2)
        muy = 2*xp.pi/self.LY * xp.arange(-self.NY/2, self.NY/2)
        muz = 2*xp.pi/self.LZ * xp.arange(-self.NZ/2, self.NZ/2)
        muExpTerm = xp.einsum('i,j,k->ijk', xp.exp(-1j*self.dt*mux**2/2.), xp.exp(-1j*self.dt*muy**2/2.), xp.exp(-1j*self.dt*muz**2/2.))
        muExpTerm = xp.fft.ifftshift(muExpTerm)
            
        potential = xp.einsum('i,j,k->ijk', xp.exp(0.5*WX**2*(x - (self.XA+self.XB)/2.)**2), xp.exp(0.5*WY**2*(y - (self.YA+self.YB)/2.)**2), 
                                  xp.exp(0.5*WZ**2*(z-(self.ZA+self.ZB)/2.)**2))
        potential = xp.log(potential) 
        
        for p in range(TIME_PTS):
            
            # Step One -- potential and interaction term
            expTerm = xp.exp(-1j* (potential + self.scatLen(p*dt)*xp.abs(solution)**2)*dt/2.0)
            solution = expTerm*solution
                        
                        
            # Step Two -- kinetic term
            fourierCoeffs = xp.fft.fftn(solution)
            fourierCoeffs = fourierCoeffs*muExpTerm
            solution = xp.fft.ifftn(fourierCoeffs)
            
            # Step Three -- potential and interaction term again
            expTerm = xp.exp(-1j* (potential + self.scatLen((p + 0.5)*dt)*xp.abs(solution)**2)*dt/2.0)
            solution = expTerm*solution
            
            # Save Solution for plotting
            #np_solution = cp.asnumpy(solution)
            psiPlot = np.vstack((psiPlot, cp.asnumpy(self.reduceIntegrate(solution)))) 
            
        return psiPlot
    
    
    # ------------------- Imaginary Time Propagation -------------------------
    
    def getEnergy(self, waveFunction):
        """
        Finds expectation value of Hamiltonian for given wave function
        """
        
        pass
    
    def normalize(self, waveFunction):
        """
        Takes unnormalized wave function and normalizes it to one
        """
        
        pass
    
    @jit(nopython=True)
    def stencil(self, solution):
        """
        Finds Laplacian of solution assuming periodic boundary conditions
        
        """
        
        pass
    
    def imagTimeProp(self, solution, tol):
        """
        
        Uses a simple forward Euler method to find ground state to within desired tolerance
        
        """
        pass
    
    def saveSolution(self, savePath)
        """
        Save the output groundstate from imaginary time propagation method
        
        """
        pass
        
    
    # -------------- Plotting, Animation, & Visualization --------------------
    
    def surfPlot(self, psiPlot):
        """
        Plots the linear z-density of the solution that resulted from realTimeProp
        
        INTPUTS:
            psiPlot -- 2D array of the integrated z density from realTimeProp
        
        """
        
        pass
    
    def animateSol(self, psiPlot, savePath=''):
        """
        This function will generate an animation of the time evolution of psi
        as given in psiPlot. This animation can be saved by passing in the 
        desired path for animation to be saved to.
        
        If you don't want to save it, don't pass in the savePath variable
        """
        
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        