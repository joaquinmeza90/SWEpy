# SWEpy: GPU-Accelerated Shallow Water Solver

**SWEpy** is a high-performance Python solver for the 2D Shallow Water Equations (SWE), utilizing the **Finite Volume Method (FVM)** on unstructured triangular meshes. It leverages **CuPy** for GPU acceleration, making it highly efficient for simulating large-scale hydraulic phenomena such as dam breaks, tsunamis, and flooding.

## 1. Key Features

*   **GPU Acceleration**: Built entirely on [CuPy](https://cupy.dev/) for massive parallel execution on NVIDIA GPUs.
*   **Robust Numerical Scheme**: Implements the **Central-Upwind** scheme (Kurganov-Petrova), ensuring:
    *   **Well-Balanced**: accurately preserves "lake-at-rest" steady states.
    *   **Positivity Preserving**: handles wetting and drying fronts (inundation) without numerical instabilities.
*   **Unstructured Meshes**: Uses triangular grids to accurately represent complex geometries and coastlines.
*   **Multiple Order Reconstructions**: Supports Constant (1st order), Linear (2nd order), and Quadratic (WENO/Limiters) spatial reconstructions.
*   **Time Integration**: Includes Forward Euler and Strong Stability Preserving Runge-Kutta (SSP-RK3) methods.

---

## 2. Installation and Environment

SWEpy relies on NVIDIA's CUDA architecture for parallel computing. To ensure stability and reproducibility, we strongly recommend using **Miniconda** to manage the Python environment.

### 2.1 Prerequisites: Hardware & Drivers
Before installing, verify your GPU compatibility. Open your terminal and run:
```bash
nvidia-smi
```
Check the `CUDA Version` in the top-right corner of the output table (e.g., 11.7 or 12.2). You need a Pascal GPU (GTX 10-series) or newer.

### 2.2 Environment Setup

#### Installing Miniconda
If you do not have Conda installed:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
```
*Note: After installation, close and reopen your terminal to initialize Conda.*

#### Cloning the Repository
```bash
git clone https://github.com/joaquinmeza90/SWEpy.git
cd SWEpy
```

#### Creating the Environment
```bash
conda create -n swepy_env python=3.10 -y
conda activate swepy_env
```

#### Installing Libraries
We provide two installation methods. **Option A (Conda)** is recommended for stability.

**Option A: Installation via Conda (Recommended)**
```bash
conda install -c conda-forge numpy scipy matplotlib cupy -y
```
*Conda will automatically detect your driver and install the correct CUDA toolkit.*

**Option B: Installation via Pip**
```bash
pip install numpy scipy matplotlib
```
Then, install CuPy matching your CUDA version (check with `nvidia-smi`):
*   For CUDA 11.x: `pip install cupy-cuda11x`
*   For CUDA 12.x: `pip install cupy-cuda12x`

### 2.3 Verifying Installation
Run the provided diagnostic script to verify that Python can talk to the GPU:

```bash
python check_installation.py
```
If successful, you will see `[✓] SUCCESS: SWEpy is ready for simulation.`

---

## 3. Usage

### 3.1 Configuration
Each simulation requires a configuration file (e.g., `Config.swe`) and a set of mesh files. An example set is provided in the `input_example/` directory.

**Example `Config.swe`**:
```text
Tmax           1000.0   # Simulation time
Dry            1e-06    # Dry depth tolerance
...
```

### 3.2 Executing the Solver
`run_sim.py` is the main execution script.

**Basic Usage (Recommendation for testing)**
To run with default settings (using `input_example` and Forward Euler):
```bash
python run_sim.py
```

**Custom Configuration**
The solver accepts command-line arguments.

*Format:*
```bash
python run_sim.py [INPUT_FOLDER] [OPTIONS]
```

*Changing the Numerical Scheme (`-m` or `--method`):*
```bash
# Use Runge-Kutta 4,3 (RK3)
python run_sim.py -m rk3
```

**Available Numerical Schemes:**
*   `fe`: Forward Euler + Minmod Reconstruction (Default)
*   `cfe`: Forward Euler + Constant Reconstruction
*   `rk3`: Runge-Kutta 4,3 + Minmod Reconstruction
*   `feweno`: Forward Euler + WENO Reconstruction
*   `rk3weno`: Runge-Kutta 4,3 + WENO Reconstruction

*Using Custom Input Data:*
```bash
python run_sim.py ./my_custom_inputs/ -m rk3weno
```

### 3.3 Visualization
Results are saved in the `Paraview/` folder created in your current working directory.
*   **Bathymetry.vtu**: Terrain mesh.
*   **Simulation_[step].vtu**: Water depth and velocities at each time step.
*   **L2Error.txt**: Error evolution (if analytic solution comparison is active).
*   **gauge_[id].txt**: Time series at specific points.

Open `.vtu` files with [Paraview](https://www.paraview.org/) to visualize.

---

## 4. Directory Structure

*   `run_sim.py`: Main driver script.
*   `check_installation.py`: Diagnostic tool.
*   `src/`: Core solver source code.
    *   `cuShallowWater.py`: Main module.
    *   `cuSolver.py`: Numerical engines.
    *   `cuFileLoader.py` / `cuFileSaver.py`: I/O handling.
*   `input_example/`: Default test case files.
*   `Paraview/`: Output directory (auto-created).

## 5. References

Based on numerical methods for Hyperbolic Conservation Laws and Shallow Water Equations, including:
*   Kurganov A., Petrova G., "A second-order well-balanced positivity preserving central-upwind scheme for the Saint-Venant system", *Communications in Mathematical Sciences*, 2007.
