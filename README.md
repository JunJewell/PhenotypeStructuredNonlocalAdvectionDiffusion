# PhenotypeStructuredNonlocalAdvectionDiffusion

Code to simulate a phenotype structured nonlocal aggregation (nonlocal advection diffusion) system, with one phenotype dimension and one physical space dimension. See the paper [[here]] for details of the model and theory. 

---

## Videos

The `Videos` folder contains animations from simulations with spatio-temporal behaviour, corresponding to figures 8 and 9 in the paper. Each video shows:


## Code Structure

### Simulation

- `parent_simulation_class.py`: Abstract base class for simulating systems with nonlocal movement and diffusion with N species/phenotype-classes. interactions.
- `nonlocal_gradient_model.py`: Daughter class of above, defining the specific form of the interactions.
- `create_phenotype_kernel_matrix.py`: Contains functions that define the phenotype interaction matrix used in `nonlocal_gradient_model.py`.

To run a simulation, create an instance of the daughter class with the desired parameters, and call either `simulate` or `simulate_with_progress` (which saves results to CSV).

### Comparison with Theory and Visualisation

- `dispersion_relation.py`: Computes the theory dispersion relations derived from linear stability analysis.
- `infer_dispersion_pydmd`: Infers the dispersion relation directly from simulation data using Dynamic Mode Decomposition and plots this against the theory dispersion relation.
- `plot_class.py`: Provides methods to visualise and save simulation results.
- `matplotlib_style.py`: Defines consistent plotting style.

### Execution Example

- `nonlocal_gradient_model_run.py`: Example script for setting the desired parameters and running a simulation.

---


## Numerical Integration
Numerical integration of integro-PDEs, such as

$$    \frac{\partial u_i(t,\mathbf{x})}{\partial t}
    =
    \nabla_{\mathbf{x}}^2u_i
    -
    \nabla_{\mathbf{x}}\cdot
    \left(
    u_i\,p(\overline{u})
    \nabla_{\mathbf{x}}
    \int_{\Omega}
    \sum_{j=1}^N
    W_{ij}(s)\,
    u_j(t,\mathbf{x}+\mathbf{s})
    \,\mathrm{d}^d\mathbf{s}
    \right)
    +
    \tilde{\beta}\sum_{j=1}^{N}L_{ij}u_j,$$

is carried out using the method-of-lines, first discretising in space and then integrating the resulting ODEs with an explicit adaptive-timestep Runge Kutta method of order 5(4). The latter is implemented through SciPy's `integrate.solve_ivp` function - see [here](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html) for details. For diffusion and advection terms, standard centred stencils are used throughout. The integral term is calculated using a fast Fourier transform method. Further detail can be found [in this repository](https://github.com/JunJewell/NonlocalReactAdvectDiffuse2D), which uses the same underlying method.
