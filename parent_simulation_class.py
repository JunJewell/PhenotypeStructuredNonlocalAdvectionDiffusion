#Parent class for N species, 1D, nonlocal advection-diffusion systems
# Class takes input of system and simulation parameters,
# and has `simulate' and `simulate_with_progress' methods outputting
# solution u_i(x,t) t, for all x in the discretised space, 
# and all t in the integration limits.
# Uses the method-of-lines to discretise integro-PDE into multiple integro-ODEs,
# integrates over time using scipy.solve_ivp, computing the integral term
# using a fast Fourier transform convolution method.
# See README.md for more details

import numpy as np
import scipy.integrate as integrator

from scipy.fftpack import fft, ifft

#Assuming:
# 1D spatial domain [0,L]
# Periodic boundary conditions
# Default initial conditions of a Gaussian perturbation about the homogeneous steady state
# (but can specify own initial conditions)
class parent_N_species_nonlocal():
    
    def __init__(self, num_species, L, kernel_name, xi, mesh_points, time_span, time_evaluations=None,
                 integrator_method="BDF",  rtol=1e-11, atol=1e-11,
                 initial_conditions=None, IC_seed=15, initial_Gaussian_std=10**(-3)):
        
        
        self.N = num_species
        
        self.L = L                   #domain is [0, L]
        
        #simulation parameters
        self.mesh_points = mesh_points #number of points to discrete in space
        self.h = self.L/self.mesh_points 
               
        self.time_span = time_span      #[start_time, end_time]
        
        
        # Save fourier transform of kernel for use in convolutions
        self.kernel_name = kernel_name
        # nonlocal signalling range
        self.xi = xi
        
        # also need number of points in kernel in order to properly roll the vector after the fourier transform
        if (kernel_name == "tophat"):
            self.fourier_kernel, self.kernel_num_points = self.fourier_tophat_kernel()
            
        elif (kernel_name == "exponential"):
            self.fourier_kernel, self.kernel_num_points = self.fourier_exponential_kernel()
        
        else:
            raise AttributeError("Please enter \"tophat\" or \"exponential\" for the kernel name.")
        
        # flag that remembers whether this simulation will be needed for dmd,
        # if it will (True), we record more time_evaluations at early time,
        # in the linear regime
        self.apply_dmd = False
        
        # which time values to store the solution.
        if (time_evaluations is None):
            if (self.time_span[1]>5 and self.time_span[0]==0):
                
                
                # Include 40*ntheta early time points to use to infer
                # the matrix governing the linear dynamics
                num_linear_sample_points = 40*self.N
                delta_t_linear_sample = 1e-3
                if (num_linear_sample_points*delta_t_linear_sample > 5.0):
                    raise AttributeError("Linear sampling range will exceed t=5.0.")
                    
                
                self.apply_dmd = True
                
                self.early_times = np.linspace(0.0 ,num_linear_sample_points*delta_t_linear_sample, num_linear_sample_points+1)
                self.intermediate_times = np.linspace((num_linear_sample_points+1)*delta_t_linear_sample, 5, 31, endpoint=False)
                self.late_times = np.linspace(5, self.time_span[1], 101)
                
                self.time_evaluations = np.concatenate([
                    self.early_times,
                    self.intermediate_times,
                    self.late_times
                    ])
            
            
            else:
                # for when we don't start near the linear regime
                self.time_evaluations = np.linspace(self.time_span[0], self.time_span[1], 101)
        else:
            # for when we specifically specify the points we want
            self.time_evaluations = time_evaluations
        
        #Using backwards differentiation formula, 'BDF', by default        
        self.integrator_method = integrator_method
        #Error tolerances for integration over time
        self.rtol = rtol
        self.atol = atol
        
        #seed for random intial conditions
        self.IC_seed = IC_seed
        self.initial_Gaussian_std = initial_Gaussian_std
        
        if (initial_conditions is None):
            #homogeneous steady state + random perturbation as ICs
            self.initial_conditions = np.zeros([self.N, self.mesh_points])
            #first add steady state
            self.initial_conditions += self.homogeneous_steady_state()[:, None]
            #add perturbation: Gaussian zero mean, 10^-3 std
            self.initial_conditions += np.random.default_rng(self.IC_seed).normal(loc=0, scale=self.initial_Gaussian_std, size = self.initial_conditions.shape)
        else:
            self.initial_conditions = initial_conditions
        
        if (self.initial_conditions[self.initial_conditions<0].size != 0):
            print("ERROR WARNING: Some initial conditions sampled with negative densities. Results will not be valid.")

    #Many of the following methods are virtual and are implemented in the specific
        # model classes, which are daughter classes of this class    

    def homogeneous_steady_state(self):
        
        raise NotImplementedError()
        
   
    def time_derivative(self, time, densities):
        
        raise NotImplementedError()
    
    #converts (N x mesh_points) array into 1D for use in solve_ivp
    def grid_to_one_dimension(self, grid):
        return grid.reshape(grid.size,)
    
    #converts the 1D state vector back to (N x mesh_points) array
    def one_dimension_to_grid(self, one_dimensional_vector):
        return one_dimensional_vector.reshape(self.N, self.mesh_points)
    
    
    #5-point stencil finite difference method with equal lengths
    # vectorised to calculate the laplacian of each species simultaneously
    def laplacian(self, u_is):
        
        laplacian_grid = np.zeros_like(u_is)
        
        #interior of laplacian using standard 3 point stencil
        laplacian_grid[:, 1:-1] = (u_is[:, 0:-2] + u_is[:, 2:] - (2*u_is[:, 1:-1])) / self.h**2
        
        #periodic BCs
        laplacian_grid[:, 0] = (u_is[:, 1] + u_is[:, -1] - (2*u_is[:, 0])) / self.h**2
        laplacian_grid[:, -1] = (u_is[:, 0] + u_is[:, -2] - (2*u_is[:, -1])) / self.h**2
        
        return laplacian_grid
    
    # input Z as a (N by meshpoints) array then calculates derivative in space
    # along the meshpoints, vectorised for each of the N species simulataneously
    def derivative_x(self, Z):
        
        deriv = np.zeros_like(Z)
        
        #using central finite difference
        deriv[:, 1:-1] = ( Z[:, 2:] - Z[:, 0:-2] ) / (2*self.h)
        #periodic BCs
        deriv[:, 0] = ( Z[:, 1] - Z[:, -1] ) / (2*self.h)
        deriv[:, -1] = ( Z[:, 0] - Z[:, -2] ) / (2*self.h)
        
        return deriv
    

    
    #Multiply a vector field ( Z_x(x,y), Z_y(x,y) ) by a 2x2 tensor/matrix
        #e.g. a rotation matrix would correspond to a chiral model
    # Probably not needed?
    def tensor_multiply(self, tensor, Z_vector):
        
        return np.tensordot(tensor, Z_vector, axes=1)

    # Returns array of shape (N, mesh_points), which is the value at each of the
    # mesh_points of the convolution of each of the N species with the kernel
    def fft_convolve_N_species_1D(self, u_is):
        # Perform FFT along axis=1 (along space for each species)
        u_fft = fft(u_is, n=self.mesh_points, axis=1)
        
        # Element-wise multiplication in frequency domain, broadcasting kernel
        conv_fft = u_fft * self.fourier_kernel[None, :]
        # Inverse FFT to return to time domain
        conv = ifft(conv_fft, axis=1).real
        # Shift the result to center the kernel correctly
        conv = np.roll(conv, -((self.kernel_num_points - 1)//2), axis=1)
        
        return conv
    

    def fourier_tophat_kernel(self):
    
        radius_length = self.xi
        number_of_points_per_radius = int(radius_length/self.h)        
      
        x = np.linspace(-radius_length, +radius_length, 1+ 2*number_of_points_per_radius)
        
        distance_from_centre = np.abs(x)
        
        #set the centre value (which is zero) to an arbitrary value to avoid
        #division by zero. This will only be used in multiplications with zero
        #so the value is not important and will always end up as zero
        distance_from_centre[number_of_points_per_radius] = -1
        
        kernel_x = np.ones_like(x)
        kernel_x[ np.where(distance_from_centre>radius_length) ] = 0
        
        #we flip the kernels as a convolution involves flipping the kernels but we just want them as they are              
        # and fourier transform for convolution
        # also record number of points in kernel in order to properly roll the fourier transform
        return (fft(np.flip(kernel_x), n=self.mesh_points), kernel_x.shape[0])
    
    #omega_0 should also equal 1 for this kernel to be properly normalised
    def fourier_exponential_kernel(self):
    
        #technically this will be missing a point on either side but they should be so close to zero it wont matter
        x = np.linspace(-self.L/2 + self.L/self.mesh_points, +self.L/2 - self.L/self.mesh_points, self.mesh_points-1)
        
        kernel_x = np.exp(-np.abs(x)/self.xi)
        
        #we flip the kernels as a convolution involves flipping the kernels but we just want them as they are              
        # and fourier transform for convolution
        # also record number of points in kernel in order to properly roll the fourier transform
        return (fft(np.flip(kernel_x), n=self.mesh_points), kernel_x.shape[0])
    
    
    
    # Only used for `direct sensing' models
     # input N array representing the interaction kernel
     # outputs the kernel multiplied by the sign of the direction 
    def multiply_kernel_by_s_hat(self, kernel):
                
        x = np.linspace(-1, +1, kernel.shape[0])
        kernel = np.sign(x)*np.flip(kernel)
        return np.flip(kernel)

    
    #Call this function to begin simulation and output the solution
    def simulate(self):
        
        solution = integrator.solve_ivp(fun=self.time_derivative,
                                        t_span=self.time_span, 
                                        y0=self.grid_to_one_dimension(self.initial_conditions),
                                        method=self.integrator_method,
                                        t_eval=self.time_evaluations,
                                        rtol=self.rtol, atol=self.atol)
        return solution


    #Same as above but saves data to textfile    
    def simulate_with_progress(self, y_file_name, t_file_name, number_of_checks=4):
        #storing every 'save_rate' datapoint
        #checks and saves progress 'number_of_checks' times
        # if self.apply_dmd is True, then first two stages will be the early
        # and intermediate time. Then the rest will be splittin up the late time
        # if self.apply_dmd is False, split evenly
        
        if (number_of_checks < 1):
            raise AttributeError("Must have 1 or more checks.")
        
        if (self.apply_dmd and number_of_checks==2):
            print("NOTE: 2 checks with a simulation for DMD will fill the first stage with early times only")
        

        if (self.apply_dmd and number_of_checks>2):
            
            if ( number_of_checks > len(self.late_times)):
                raise AttributeError("Need less checks, otherwise late_times will be split into empty arrays.")
            
            t_eval_sub_arrays = np.array(
                [self.early_times,
                 self.intermediate_times,
                 *np.array_split(self.late_times, number_of_checks - 2)],
                dtype=object
            )
            
        else:
            t_eval_sub_arrays = np.array_split(self.time_evaluations, number_of_checks) 
            
            
        
        solution = integrator.solve_ivp(fun=self.time_derivative,
                                        t_span=(self.time_span[0],  t_eval_sub_arrays[0][-1]), 
                                        y0=self.grid_to_one_dimension(self.initial_conditions),
                                        method=self.integrator_method,
                                        t_eval= t_eval_sub_arrays[0],
                                        rtol=self.rtol, atol=self.atol)
        
        if (solution.success == False):
            print("ERROR: Integration failed during the 1st stage.")
            return solution
        

        
        np.savetxt(y_file_name, solution.y, delimiter=',')
        np.savetxt(t_file_name, solution.t, delimiter=',')
        
        next_solution = solution
        for i in range( 1, len(t_eval_sub_arrays) ):
            
            print("%d/%d stages complete" %(i, number_of_checks)  )
            
            next_solution = integrator.solve_ivp(fun=self.time_derivative,
                                            t_span=(t_eval_sub_arrays[i-1][-1],  t_eval_sub_arrays[i][-1]), 
                                            y0=self.grid_to_one_dimension(next_solution.y[:,-1]),
                                            method=self.integrator_method,
                                            t_eval= t_eval_sub_arrays[i],
                                            rtol=self.rtol, atol=self.atol)
            
            #concatenate the next solution with the previous solution
            #Note: leaving "sol" attribute unchanged as assuming dense_output=False
            solution.t=np.concatenate((solution.t, next_solution.t))
            solution.y=np.concatenate((solution.y, next_solution.y), 1)
            
            if(solution.t_events is not None and next_solution.t_events is not None):
                solution.t_events=np.concatenate((solution.t_events, next_solution.t_events))
            if(solution.y_events is not None and next_solution.y_events is not None):    
                solution.y_events=np.concatenate((solution.y_events, next_solution.y_events))
            
            solution.nfev=solution.nfev + next_solution.nfev
            solution.njev=solution.njev + next_solution.njev
            solution.nlu=solution.nlu + next_solution.nlu
            solution.status=next_solution.status
            solution.message=next_solution.message
            solution.success=next_solution.success
            
            
            if(next_solution.success == False):
                print("ERROR: Integration failed during stage %d/%d ." %(i+1, number_of_checks))
                return solution
            
            #update data file
            np.savetxt(y_file_name, solution.y, delimiter=',')
            np.savetxt(t_file_name, solution.t, delimiter=',')
            
        print("100% complete")
        return solution
    