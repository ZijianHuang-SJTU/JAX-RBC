# JAX-RBC

**JAX-RBC** is a GPU-accelerated solver for simulating three-dimensional Rayleigh–Bénard convection in a closed cavity. The solver is implemented in Python with JAX and combines high-order finite-difference schemes, stretched Cartesian grids, and single-host multi-GPU parallelism.

## Features

- Multi-GPU parallelization with JAX `pmap` and z-direction domain decomposition
- High-order upwind schemes for convective fluxes and central schemes for viscous and thermal diffusion
- Supports stretched grids, restart simulations, and online heat-transfer and turbulence statistics

## Included Codes

The repository contains three codes for different simulation tasks:

- `RB_convection_2D.py`: Two-dimensional Rayleigh–Bénard convection solver
- `RB_convection_3D.py`: Three-dimensional Rayleigh–Bénard convection solver
- `RB_convection_C.py`: Differentiable thermal control for Rayleigh–Bénard convection

## How to Run

The codes are written in Python using JAX and can be opened and launched from Visual Studio Code.

1. Clone the repository and open the project folder in VS Code:

   ```bash
   git clone https://github.com/ZijianHuang-SJTU/JAX-RBC.git
   cd JAX-RBC
   ```

2. In VS Code, select **Python: Select Interpreter** from the Command Palette and install the required packages:

   ```bash
   python -m pip install numpy scipy
   ```

   Install JAX for your CPU or GPU platform by following the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

3. Modify the physical and numerical parameters near the beginning of the selected source file. From the repository root, run one of the following commands:

   ```bash
   python RB_convection_2D.py
   python RB_convection_3D.py
   python RB_convection_C.py
   ```

Alternatively, open a source file in VS Code and click **Run Python File** in the upper-right corner.

> **Note:** Multi-GPU simulations require a compatible GPU-enabled JAX environment and enough visible devices for the configured `num_dev` value.
