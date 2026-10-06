import numpy as np

# theory dispersion relation class
    # input of model parameters and array of k values:
        # floats: U0, D, mu, B_kernel, C_kernel, spatial_kernel, xi, beta
        # 1D array of k values
    # 4 methods to calculate lambda  
        # lambda, when beta=0 limit equation
        # lambda, when beta->infinity limit equation
        # lambda, when beta generic, and B=1-cos(pi theta), C=1-cos(pi theta)
        # lambda, when beta generic, and B=1-cos(pi theta), C=1+cos(pi theta)        
            
class dispersion():
    
    def __init__(self, k, U, D, mu, W_matrix, spatial_kernel_name, xi, beta):
        
        
        self.k = k   
        self.U = U
        self.D = D
        self.mu = mu
        
        self.W_matrix = W_matrix    # W(theta, phi)=mu*B(theta)C(phi)
        
        self.ntheta = np.shape(W_matrix)[0]
        if ( W_matrix.shape != (self.ntheta, self.ntheta)):
            raise AttributeError("W matrix should be an ntheta x ntheta array because W(theta, phi)=mu*B(theta)C(phi).")
        
        self.spatial_kernel_name = spatial_kernel_name
        self.xi = xi
        self.beta = beta
        
        # k_squared_G_tilde_k = k^2* ( U*p'(U)*fourier_transform(spatial_kernel) )
        # volume-filling has been scaled so total volume filling is 1-U
        self.k_squared_G_tilde_k = (self.k**2) * self.U * (1-self.U)
        
        if ( spatial_kernel_name == "tophat" ):
            self.k_squared_G_tilde_k *= 2*np.sin(self.k*self.xi)/(self.k+1e-13)
        elif ( spatial_kernel_name == "exponential" ):
            self.k_squared_G_tilde_k *= 2*self.xi/(1 + np.power(self.k*self.xi,2))
        else:
            raise AttributeError("Please enter \"tophat\" or \"exponential\" for the spatial kernel name.")
            
    
    # predicted lambda in the limit beta->0 for generic phenotype kernels B C
    def lambda_beta_0_limit(self):
        # depends on average (total) of product B and C
        average_of_product_B_C = np.trace(self.W_matrix)/self.ntheta
        return -(self.D*self.k**2) + self.k_squared_G_tilde_k* average_of_product_B_C
    
    
    # predicted lambda in the limit beta->infinity for generic phenotype kernels B C
    def lambda_beta_infinity_limit(self):
        # depends on product of average (total) of B and C
        product_of_average_B_C = np.sum(self.W_matrix)/(self.ntheta**2)
        return -(self.D*self.k**2) + self.k_squared_G_tilde_k* product_of_average_B_C
    

    def lambda_sin_sin_exact(self):
        A = self.mu * self.k_squared_G_tilde_k
        rad = 9.0 * A**2 + 4.0 * self.beta * np.pi**2 * A + 4.0 * self.beta**2 * np.pi**4
        
        return (
            -(self.D * self.k**2)
            + (3.0 / 4.0) * A
            - 0.5 * self.beta * np.pi**2
            + (1.0 / 16.0) * np.sqrt(rad)
        )

    def lambda_sin_minus_sin_exact(self):
        A = self.mu * self.k_squared_G_tilde_k
        rad = 1.0 * A**2 + 12.0 * self.beta * np.pi**2 * A + 4.0 * self.beta**2 * np.pi**4
        
        return (
            -(self.D * self.k**2)
            + (1.0 / 4.0) * A
            - 0.5 * self.beta * np.pi**2
            + (1.0 / 4.0) * np.sqrt(rad)
        )
