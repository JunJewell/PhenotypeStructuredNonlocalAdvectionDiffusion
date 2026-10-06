"""
Inferring linear growth rates, \lambda(k), directly from time series data of
the simulation. No knowledge of the parameters (other than the geometry) is 
given a priori. This is a valuable test for linear stability analysis of pattern
formation.

It works by applying Exact Dynamic Mode Decomposition independetly to each
Fourier mode (in physical space).
"""

import matplotlib.pyplot as plt
import numpy as np

# REQUIRES pydmd package
from pydmd import DMD


# time series data comes as a 1D array,
# so needs reshaping to 3D (phenotype x space x time) array
def reshape_solution(y, n_theta, nx, nt):
    y = np.asarray(y)
    if y.shape == (n_theta * nx, nt):
        return y.reshape(n_theta, nx, nt)
    if y.shape == (n_theta, nx, nt):
        return y
    if y.shape == (nx, n_theta, nt):
        return y.transpose(1, 0, 2)
    raise ValueError(f"Unexpected solution shape: {y.shape}")



def infer_dispersion_dmd(
    t,                  # array of time values
    y,                  # 1D array of numerical solution values (to be reshaped)
    n_theta,            # number of points for discretisation in phenotype space
    nx,                 # number of points for discretisation in physical space
    L_x,                # domain length physical space
    t_start,            # start of time window for applying DMD
    t_end,              # end of time window for applying DMD
    U=0.0,              # homogeneous state being linearised about
    svd_rank=0.9999,    # parameter for truncation of snapshot matrix, 0.9999 used when data is noisy / poorly conditioned
    k_indices=None,
    k_max=None
):
    
    t = np.asarray(t, dtype=float).reshape(-1)
    u = reshape_solution(y, n_theta, nx, t.size).astype(float, copy=True)


    # check the U parameter supplied is a scalar for all phenotypes not a vector
    if np.isscalar(U):
        u -= float(U)
    else:
        U = np.asarray(U, dtype=float).reshape(-1)
        if U.size != n_theta:
            raise ValueError("U must be scalar or have length n_theta")
        u -= U[:, None, None]

    indices = np.flatnonzero((t >= t_start) & (t <= t_end))
    if indices.size < 3:
        raise ValueError("The inference window must contain at least 3 snapshots")

    times = t[indices]
    steps = np.diff(times)
    dt = float(np.median(steps))
    if not np.allclose(steps, dt, rtol=1e-6, atol=1e-12):
        raise ValueError("Snapshot times must be equally spaced")

    # Fourier coefficients: (phenotype field, spatial Fourier bin, time)
    u_hat = np.fft.rfft(u[:, :, indices], axis=1)

    if k_indices is None:
        if k_max is None:
            k_indices = np.arange(1, u_hat.shape[1])
        else:
            max_bin = int(np.floor(k_max * L_x / (2.0 * np.pi)))
            max_bin = min(max_bin, u_hat.shape[1] - 1)
            k_indices = np.arange(1, max_bin + 1)
    else:
        k_indices = np.asarray(k_indices, dtype=int)
        if np.any((k_indices < 1) | (k_indices >= u_hat.shape[1])):
            raise ValueError("Invalid k_indices")

    k = 2.0 * np.pi * k_indices / L_x
    lambdas = []
    modes = []

    # This is the key logic that does the DMD indepedently for each mode
    for k_index in k_indices:
        snapshots = u_hat[:, k_index, :]

        if np.linalg.norm(snapshots) <= np.finfo(float).eps:
            lambdas.append(np.empty(0, dtype=complex))
            modes.append(np.empty((n_theta, 0), dtype=complex))
            continue

        dmd = DMD(svd_rank=svd_rank, exact=True).fit(snapshots)
        mu = np.asarray(dmd.eigs)
        keep = np.abs(mu) > np.finfo(float).eps

        lambdas.append(np.log(mu[keep]) / dt)
        modes.append(np.asarray(dmd.modes)[:, keep])

    return k, lambdas, modes, dt


def dominant_growth_rates(lambdas):
    return np.array([
        np.max(values.real) if values.size else np.nan
        for values in lambdas
    ])





if __name__ == "__main__":
    from dispersion_relation import dispersion
    import create_phenotype_kernel_matrix as cpkm
    
    import matplotlib
    from matplotlib_style import matplotlib_style
    
    plt.rcParams.update(matplotlib_style(**{"text.usetex":True})) 
    matplotlib.use('pgf')

    folder_tag = "results/"
    folder_tag += "example_1"
    
    y_file_name = folder_tag + "/data_u_theta.csv"
    t_file_name = folder_tag + "/data_t.csv"
    parameter_file_name = folder_tag + "/parameters.txt"
    
    #Reading in parameters
    print("Reading in parameters...")
    with open(parameter_file_name) as param_file:
        parameters_string = param_file.read()

    #All the values we need are between '=' and '\n'
    #change all '\n' to '='
    parameters_string = parameters_string.replace('\n','=')
    #remove all white space
    parameters_string = ''.join(parameters_string.split())
    #take everything in between the '='
    parameters_list = parameters_string.split(sep='=')
    #only take every other element so we only take parameter values not names
    parameters_list = parameters_list[1::2]
    

    n_theta = int(parameters_list[0])
    
    
    nx = int(parameters_list[10])
    L_x = float(parameters_list[9])
    U = float(parameters_list[5])

    D = float(parameters_list[6])
    xi = float(parameters_list[7])
    mu = float(parameters_list[1])
    beta = float(parameters_list[2])
    kernel_name = str(parameters_list[8])
    
    B_kernel_name = str(parameters_list[3])
    C_kernel_name = str(parameters_list[4])
    
    
    W = mu*cpkm.build_separable_matrix( n_theta, B_kernel_name , C_kernel_name )
    
    print("Loading solution data...")
    solution_y = np.loadtxt(y_file_name, delimiter=",")
    solution_t = np.loadtxt(t_file_name, delimiter=",")
    
    
    
    # time window to infer growth rates (choose near start for linear regime)
    # window can be smaller for less phenotypes
    t_start = 0.1
    t_end = t_start + n_theta*6e-3
    
    k_nyquist = np.pi * 256 / 10    # highest possible k the meshc could ever resolve
    k_max = 0.4 * k_nyquist

    k, lambdas, modes, dt = infer_dispersion_dmd(
        solution_t,
        solution_y,
        n_theta=n_theta,
        nx=nx,
        L_x=L_x,
        t_start=t_start,
        t_end=t_end,
        U=U,
        k_max = k_max
    )

    dominant = dominant_growth_rates(lambdas)

    

    
    
    
    theory_k = np.linspace(0.01, 100.0, 10_000)
    theory = dispersion(theory_k, U, D, mu, W, kernel_name, xi, beta)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(theory_k, theory.lambda_beta_0_limit(), label=r"Theory, $\beta\to 0$", linewidth=4.)
    ax.plot(theory_k, theory.lambda_beta_infinity_limit(), label=r"Theory, $\beta\to \infty$", linewidth=4.)
    ax.plot(theory_k, theory.lambda_sin_minus_sin_exact(), label=r"Theory, $\beta$ exact", linewidth=4.)



    ax.scatter(k, dominant, label="Simulation", marker='.', c='r', s=500,zorder=10)
    ax.axhline(0.0, linestyle="--", linewidth=2., c='black')
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\lambda(k)$")
    ax.set_xlim(0,6)
    ax.set_ylim(-30, 10)
    ax.set_yticks([-30, -20, -10, 0, 10])
    ax.set_xticks([0, 2, 4, 6])
    ax.set_box_aspect(0.9295303054)
    ax.legend(markerscale=1, )
    
    ax.minorticks_off()
    
    plt.tight_layout()
    fig.savefig(folder_tag +"/dispersion.pdf", bbox_inches="tight")

    print(f"DMD timestep: {dt:.6g}")
    for wavenumber, growth in zip(k, dominant):
        print(f"k={wavenumber:8.4f}  max Re(lambda)={growth: .6e}")