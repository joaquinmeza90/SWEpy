
# SWEpy: GPU-Accelerated Shallow Water Solver

**SWEpy** is a high-performance Python solver for the 2D Shallow Water Equations (SWE), utilizing the **Finite Volume Method (FVM)** on unstructured triangular meshes. It leverages **CuPy** for GPU acceleration, making it highly efficient for simulating large-scale hydraulic phenomena such as dam breaks, tsunamis, and flooding.

## Key Features

*   **GPU Acceleration**: Built entirely on [CuPy](https://cupy.dev/) for massive parallel execution on NVIDIA GPUs.
*   **Robust Numerical Scheme**: Implements the **Central-Upwind** scheme (Kurganov-Petrova), ensuring:
    *   **Well-Balanced**: accurately preserves "lake-at-rest" steady states.
    *   **Positivity Preserving**: handles wetting and drying fronts (inundation) without numerical instabilities.
*   **Unstructured Meshes**: Uses triangular grids to accurately represent complex geometries and coastlines.
*   **Multiple Order Reconstructions**: Supports Constant (1st order), Linear (2nd order), and Quadratic (WENO/Limiters) spatial reconstructions.
*   **Time Integration**: Includes Forward Euler and Strong Stability Preserving Runge-Kutta (SSP-RK3) methods.

## Dependencies

*   **Python** 3.x
*   **CuPy** (Requires strictly NVIDIA GPU + CUDA)
*   **NumPy**
*   **SciPy**

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/joaquinmeza90/SWEpy.git
    cd SWEpy
    ```

2.  Install dependencies. Using `conda` is recommended for handling CUDA packages:
    ```bash
    conda install -c conda-forge cupy scipy numpy
    ```
    *Note: Ensure your CUDA version matches the CuPy package.*

## Usage

### 1. Configuration
Each simulation requires a configuration file (e.g., `Config.swe`) and a set of mesh files (Coordinates, Connectivity, Neighbors, etc.). See the `Examples/` directory for structure.

**Example `Config.swe`**:
```
Tmax           1000.0   # Simulation time
Dry            1e-06    # Dry depth tolerance
Gravity        9.81
CFL            0.5      # Courant–Friedrichs–Lewy number
dt_save        1.0      # Output interval
...
```

### 2. Running a Simulation
Create a Python script to load the case and execute the solver.

```python
import sys
import os

# Add the source directory to the system path
sys.path.append("src")

import cuFileLoader as FileLoader
import cuShallowWater as swe

# Path to the directory containing Config.swe and mesh files
case_directory = "Examples/Malpasset"

# 1. Load the mesh and configuration
print(f"Loading case from {case_directory}...")
mesh = FileLoader.load_from_files(case_directory)

# 2. Run the simulation
# Options for ts_option: 'RK3' (Recommended), 'FE' (Forward Euler), 'RK3WENO' (High order)
print("Starting simulation...")
swe.run(mesh, ts_option="RK3")
```

### 3. Visualization
Simulation results are saved in a `Paraview/` subdirectory within the case folder. The files are in **VTK Unstructured Grid (.vtu)** format.
*   **Solution.[t].vtu**: Contains Water Depth, Velocities, and Discharges.
*   **Bathymetry.vtu**: Contains the terrain/bathymetry mesh.

Open these files with [Paraview](https://www.paraview.org/) to visualize the results.

## Directory Structure

*   `src/`: Core solver source code.
    *   `cuSolver.py`: Main numerical engine.
    *   `cuCentralUpwindMethod.py`: Flux and source term calculations.
    *   `cuFileLoader.py`: Mesh and input I/O.
    *   `cuFileSaver.py`: VTK output generation.
*   `Examples/`: Benchmark test cases (e.g., Malpasset Dam Break, Circular Dam Break).

## References

Based on numerical methods for Hyperbolic Conservation Laws and Shallow Water Equations, including:
*   Kurganov A., Petrova G., "A second-order well-balanced positivity preserving central-upwind scheme for the Saint-Venant system", *Communications in Mathematical Sciences*, 2007.
