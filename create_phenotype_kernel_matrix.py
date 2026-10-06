import numpy as np

def theta_grid(ntheta):
    """
    Construct a cell-centred phenotype grid on [0,1].

    Parameters
    ----------
    ntheta : int
        Number of phenotype bins.

    Returns
    -------
    theta : ndarray, shape (ntheta,)
        Cell-centred grid points:
        theta_j = (j + 0.5) / ntheta, j = 0,...,ntheta-1
    """
    if ntheta <= 0:
        raise ValueError("ntheta must be a positive integer.")
    return (np.arange(ntheta) + 0.5) / ntheta


def separable_W_matrix(theta, B_func, C_func):
    """
    Construct W_{jk} = B(theta_j) * C(theta_k)

    Parameters
    ----------
    theta : array_like, shape (ntheta,)
        Phenotype grid (cell centers recommended).
    B_func : callable
        Function B(theta): receiver sensitivity.
    C_func : callable
        Function C(theta): sender signal strength.

    Returns
    -------
    W : ndarray, shape (ntheta, ntheta)
        Separable interaction matrix.
    """
    B = B_func(theta)          # shape (ntheta,)
    C = C_func(theta)          # shape (ntheta,)
    return np.outer(B, C)


def linear_increasing(theta):
    """
    Linear, monotonic increasing on [0,1]:
    f(0)=0, f(1)=1.
    Accepts scalar or array-like theta in [0,1].
    """
    th = np.asarray(theta)
    out = th.copy()
    # numerical safety: clip to [0,1] then ensure exact endpoints
    out = np.clip(out, 0.0, 1.0)
    out = out.astype(float)
    out[...][np.isclose(out, 0.0)] = 0.0
    out[...][np.isclose(out, 1.0)] = 1.0
    return out

def linear_decreasing(theta):
    """
    Linear, monotonic decreasing on [0,1]:
    f(0)=1, f(1)=0.
    """
    return 1.0 - linear_increasing(theta)

def sine_squared_halfpi(theta):
    """
    Smooth monotonic increasing function:
    f(theta) = 2*sin(pi*theta/2)^2, so f(0)=0, f(1)=2.
    This is smooth and strictly increasing on [0,1].
    """
    th = np.asarray(theta)
    th = np.clip(th, 0.0, 1.0)
    return 2*np.sin(0.5 * np.pi * th)**2

def sine_squared_halfpi_decreasing(theta):
    """
    Smooth monotonic decreasing function:
    f(theta) = 2. - sin(pi*theta/2)^2, so f(0)=2, f(1)=0.
    This is smooth and strictly decreasing on [0,1].
    """
    
    return 2.0 - sine_squared_halfpi(theta)


def sine_squared_halfpi_with_third_negative(theta):
    """
    Smooth monotonic increasing function:
    f(theta) = 6*sin(pi*theta/2)^2 - 2, so f(0)=-2, f(1)=4.
    This is smooth and strictly increasing on [0,1].
    It has some negative region to demonstrate a repulsive-attractive mix.
    """

    return 6*sine_squared_halfpi(theta) - 2



def build_separable_matrix( n_theta, B_name , C_name ):
    """
    Parameters
    ----------
    n_theta : int
        number of discretisations of the phenotype space
    B_name : str
        name to choose B kernel (function)
    C_name : str
        name to choose C kernel (function)

    Returns a separable phenotype kernel matrix W_ij = B_i C_j
    -------
    None.

    """
    
    thetas = theta_grid(n_theta)
    
    names = (B_name, C_name)
    
    # array which will be [B_function, C_function]
    functions = []
    
    for i in range(len(names)):
        
        if ( names[i] == "sine+" ):
            functions.append(sine_squared_halfpi)
            
        elif ( names[i] == "sine-" ):
            functions.append(sine_squared_halfpi_decreasing)
            
        elif ( names[i] == "sine+-" ):
            functions.append(sine_squared_halfpi_with_third_negative)
            
        elif ( names[i] == "linear+" ):
            functions.append(linear_increasing)
            
        elif ( names[i] == "linear-" ):
            functions.append(linear_decreasing)
        
        else:
            raise ValueError("Phenotype kernel names must be one of [sine+, sine-, sine+-, linear+, linear-]")
    
    return separable_W_matrix(thetas, functions[0], functions[1])


def distance_kernel_matrix_physical(theta, kernel_func, periodic=True, dtype=float):
    """
    Build A[i, j] = kernel_func(|theta_i - theta_j|).

    Parameters
    ----------
    theta : array_like, shape (n,)
        Phenotype grid points.
    kernel_func : callable
        Function of a nonnegative distance r.
    periodic : bool
        If True, use wrap-around distance on [0,1]:
            r = min(|theta_i - theta_j|, 1 - |theta_i - theta_j|)
    dtype : data-type
        Output dtype.

    Returns
    -------
    A : ndarray, shape (n, n)
        Distance-based interaction matrix.
    """
    theta = np.asarray(theta, dtype=float)
    if theta.ndim != 1:
        raise ValueError("theta must be a 1D array.")

    dist = np.abs(theta[:, None] - theta[None, :])

    if periodic:
        dist = np.minimum(dist, 1.0 - dist)

    A = np.vectorize(kernel_func, otypes=[dtype])(dist)
    return A.astype(dtype)


def exponential_decay_kernel(xi):
    """
    Return a kernel k(r) = exp(-r / xi), where r is a physical distance.

    Parameters
    ----------
    xi : float
        Decay length scale. Must be positive.
    """
    if xi <= 0:
        raise ValueError("xi must be positive.")

    def k(r):
        r = np.asarray(r, dtype=float)
        return np.exp(-r / xi)

    return k

def two_lengthscale_exponential_kernel(B, xi_1, xi_2):
    """
    Return a two-lengthscale kernel

        k(r) = exp(-r^2 / (2*xi_1^2))
             - B * exp(-r^2 / (2*xi_2^2)),

    where r is a physical distance.

    Typically choose xi_2 > xi_1 and 0 < B < 1 so that the kernel
    starts positive, becomes negative, and approaches zero from below.

    Parameters
    ----------
    B : float
        Strength of the longer-range negative interaction.
    xi_1 : float
        Short interaction length scale. Must be positive.
    xi_2 : float
        Long interaction length scale. Must be positive.
    """
    if xi_1 <= 0 or xi_2 <= 0:
        raise ValueError("Length scales must be positive.")

    if not 0 < B < 1:
        raise ValueError("B must satisfy 0 < B < 1.")

    if xi_2 <= xi_1:
        raise ValueError("xi_2 must be greater than xi_1.")

    def k(r):
        r = np.asarray(r, dtype=float)
        return (
            np.exp(-r**2 / (2 * xi_1**2))
            - B * np.exp(-r**2 / (2 * xi_2**2))
        )

    return k


def gaussian_decay_kernel(xi):
    """
    Return a kernel k(r) = exp(-(r / xi)^2), where r is a physical distance.
    """
    if xi <= 0:
        raise ValueError("xi must be positive.")

    def k(r):
        r = np.asarray(r, dtype=float)
        return np.exp(-(r / xi) ** 2)

    return k


def power_law_decay_kernel(alpha, eps=1e-12):
    """
    Return a kernel k(r) = 1 / (eps + r)^alpha, where r is a physical distance.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if eps <= 0:
        raise ValueError("eps must be positive.")

    def k(r):
        r = np.asarray(r, dtype=float)
        return 1.0 / (eps + r) ** alpha

    return k


def exponential_decay_matrix(theta, xi, periodic=True, dtype=float):
    """
    Convenience wrapper for A[i, j] = exp(-|theta_i - theta_j| / xi).
    """
    return distance_kernel_matrix_physical(
        theta,
        kernel_func=exponential_decay_kernel(xi),
        periodic=periodic,
        dtype=dtype,
    )


def gaussian_decay_matrix(theta, xi, periodic=True, dtype=float):
    """
    Convenience wrapper for A[i, j] = exp(-(|theta_i - theta_j| / xi)^2).
    """
    return distance_kernel_matrix_physical(
        theta,
        kernel_func=gaussian_decay_kernel(xi),
        periodic=periodic,
        dtype=dtype,
    )

