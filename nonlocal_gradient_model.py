# Phenotype structured model

from parent_simulation_class import parent_N_species_nonlocal
import numpy as np

class Phenotype_nonlocal_gradient_with_volumefilling_model(parent_N_species_nonlocal):
    """
    u is stored as an array shape (ntheta, mesh_points),
    keeping the same orientation as the parent class (originally (N, mesh_points)).
    """

    def __init__(self, U_theta, D_theta, W_matrix, ntheta, L,
                 kernel_name, xi, mesh_points, time_span, beta=0.0, no_flux_BC=True,
                 time_evaluations=None, integrator_method="BDF", rtol=1e-11, atol=1e-11,
                 initial_conditions=None, IC_seed=15, initial_Gaussian_std=1e-3):

        # store number of phenotype bins
        self.ntheta = ntheta
        
        self.U_theta = U_theta.copy()          # homogeneous steady state across theta

        # physical domain setup delegated to parent (keeps mesh_points, h, convolution, etc)
        # we call parent's constructor with num_species = ntheta so internal shapes match
        super().__init__(self.ntheta, L, kernel_name, xi, mesh_points, time_span,
                         time_evaluations=time_evaluations, integrator_method=integrator_method,
                         rtol=rtol, atol=atol, initial_conditions=initial_conditions,
                         IC_seed=IC_seed, initial_Gaussian_std=initial_Gaussian_std)

        # check shapes
        if (U_theta.shape != (ntheta,) or D_theta.shape != (ntheta,)
            or W_matrix.shape != (ntheta, ntheta)):
            raise AttributeError("Please input vectors/matrices consistent with ntheta.")


        self.D_theta = D_theta.copy()          # spatial diffusivities (per theta-bin)
        self.W = W_matrix.copy()               # W[j,k] = interaction weight from phi=k to theta=j
        self.dtheta = 1.0/self.ntheta          # V, the phenotype space, is defined as [0,1]
        self.beta = beta                       # phenotype diffusion coefficient
        
        self.no_flux_BC = no_flux_BC    # Boolean for whether to use no flux BCs (True) or periodic (False)

        # if you prefer uniform bins and different definition, overwrite self.dtheta before simulating

    def homogeneous_steady_state(self):
        # Return vector of length ntheta (steady-state per theta)
        return self.U_theta

    def laplacian_theta_no_flux(self, u_is):
        
        if (self.ntheta == 1):
            return np.zeros_like(u_is)
        
        laplacian_grid = np.zeros_like(u_is)
    
        # interior points
        laplacian_grid[1:-1, :] = (u_is[0:-2, :] + u_is[2:, :] - 2.0*u_is[1:-1, :]) / self.dtheta**2
    
        # left boundary (Neumann: u_{-1} = u_0)
        laplacian_grid[0, :] = (1.0*u_is[1, :] - 1.0*u_is[0, :]) / self.dtheta**2
    
        # right boundary (Neumann: u_{N} = u_{N-1})
        laplacian_grid[-1, :] = (1.0*u_is[-2, :] - 1.0*u_is[-1, :]) / self.dtheta**2
    
        return laplacian_grid

    def laplacian_theta_periodic(self, u_is):
        
        if (self.ntheta == 1):
            return np.zeros_like(u_is)
        
        laplacian_grid = np.zeros_like(u_is)
    
        # interior points
        laplacian_grid[1:-1, :] = (u_is[0:-2, :] + u_is[2:, :] - 2.0*u_is[1:-1, :]) / self.dtheta**2


        #periodic BCs
        laplacian_grid[0, :] = (u_is[1, :] + u_is[-1, :] - (2.0*u_is[0, :])) / self.dtheta**2
        laplacian_grid[-1, :] = (u_is[0, :] + u_is[-2, :] - (2.0*u_is[-1, :])) / self.dtheta**2
    
        return laplacian_grid


    def time_derivative(self, time, densities):
        """
        densities: 1D vector shape (ntheta * mesh_points,)
        Return flattened derivative with same ordering as parent (ntheta, mesh_points).
        """
    
        # reshape into (ntheta, mesh_points)
        u_theta = self.one_dimension_to_grid(densities)
        
        if (self.no_flux_BC):
            laplacian_theta = self.laplacian_theta_no_flux(u_theta)
        else:
            laplacian_theta = self.laplacian_theta_periodic(u_theta)
    
        u_theta_derivative = (
            # spatial diffusion
            self.D_theta[:, None] * self.laplacian(u_theta)
    
            # minus advective nonlocal flux divergence
            - self.derivative_x(
                u_theta
                * (1.0 - np.sum(u_theta, axis=0)* self.dtheta)[None, :]
                * self.derivative_x(
                    np.matmul(
                        self.W,
                        self.fft_convolve_N_species_1D(u_theta)*self.h
                    ) * self.dtheta
                )
            )
    
            # plus phenotype diffusion
            + self.beta * laplacian_theta
        )
    
        return self.grid_to_one_dimension(u_theta_derivative)

