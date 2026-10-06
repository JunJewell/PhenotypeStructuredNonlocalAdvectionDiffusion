import numpy as np
import os
import create_phenotype_kernel_matrix as cpkm
from nonlocal_gradient_model import Phenotype_nonlocal_gradient_with_volumefilling_model

# output folder
folder_tag = "results/"
folder_tag += "example_1"
os.makedirs(folder_tag, exist_ok=False)



# PARAMETERS
ntheta = 50    # number of phenotype bins
mu = 50.0
beta = 0.0001  # phenotype diffusion coefficient
U0 = 0.1 * np.ones(ntheta)  # mass per unit theta per unit x, homogeneous steady state
D_theta = 1.0 * np.ones(ntheta)
xi = 1.0
kernel_name = "exponential"
L = 10
mesh_points = 256
time_span = (0, 500)
IC_seed = 15
no_flux_BC = True   # Boolean for whether to use no flux BCs (True) or periodic (False)

integrator_method="RK45"

time_evaluations = None
#time_evaluations = np.linspace(time_span[0], time_span[1], 1001)


# construct W matrix (ntheta x ntheta). W_matrix  = B(theta)*C(\phi)
B_kernel_name = "sine+"
C_kernel_name = "sine+"
W = cpkm.build_separable_matrix( ntheta, B_kernel_name , C_kernel_name )





#save text file specifying all the parameters
parameter_string = ("ntheta=%d \n"
                    "mu=%.10f \n"
                    "beta=%.10f \n"
                    "B_kernel=%s \n"
                    "C_kernel=%s \n"
                    "U=%.10f \n"
                    "D=%.10f \n"
                    "xi=%.10f \n"
                    "kernel=%s \n"
                    "L=%.10f \n"
                    "mesh_points=%d \n"
                    "initial_time=%.10f \n"
                    "final_time=%.10f \n"
                    "initial_conditions_seed=%d \n" 
                    "no_flux_BC=%i"
                    %(ntheta, mu, beta, B_kernel_name, C_kernel_name, U0[0], D_theta[0], xi,
                      kernel_name, L, mesh_points, time_span[0], time_span[1],
                      IC_seed, no_flux_BC)
                    )


with open("%s/parameters.txt" %folder_tag, "w") as text_file:
    text_file.write(parameter_string)

# create simulation object
system = Phenotype_nonlocal_gradient_with_volumefilling_model(
    U_theta=U0,
    D_theta=D_theta,
    W_matrix=W,
    ntheta=ntheta,
    L=L,
    kernel_name=kernel_name,
    xi=xi,
    mesh_points=mesh_points,
    time_span=time_span,
    beta=beta,
    no_flux_BC = no_flux_BC,
    IC_seed=IC_seed,
    initial_conditions=None,
    integrator_method=integrator_method,
    time_evaluations = time_evaluations
)

# run and save
print("Beginning phenotype-structured simulation")
solution = system.simulate_with_progress("%s/data_u_theta.csv" % folder_tag, "%s/data_t.csv" % folder_tag)
print("Finished")
