# JAX-RBC

**JAX-RBC** is a GPU-accelerated solver for simulating three-dimensional Rayleigh–Bénard convection in a closed cavity. The solver is implemented in Python with JAX and combines high-order finite-difference schemes, stretched Cartesian grids, and single-host multi-GPU parallelism.

## Features

- Multi-GPU parallelization with JAX `pmap` and z-direction domain decomposition
- High-order upwind schemes for convective fluxes and central schemes for viscous and thermal diffusion
- Supports stretched grids, restart simulations, and online heat-transfer and turbulence statistics
- 
## Included Codes

The repository contains three codes for different simulation tasks:
- `RB_convection_2D`: Two-dimensional Rayleigh–Bénard convection solver
- `RB_convection_3D`: Three-dimensional Rayleigh–Bénard convection solver
- `RB_convection_C`: Differentiable thermal control for Rayleigh–Bénard convection
