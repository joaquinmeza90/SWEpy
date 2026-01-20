import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import cuShallowWater as SWE


import argparse

def main():
    parser = argparse.ArgumentParser(description='SWEpy: GPU Accelerated Shallow Water Equation Solver')
    parser.add_argument('input_folder', nargs='?', default='./input_example/', help='Path to folder containing input files (default: ./input_example/)')
    parser.add_argument('--method', '-m', default='fe', 
                        choices=['fe', 'cfe', 'rk3', 'feweno', 'rk3weno'], 
                        help='Time integration and reconstruction method (default: fe)')

    args = parser.parse_args()

    # Normalize path
    path2load = os.path.join(args.input_folder, '') # ensure trailing slash

    print(f"Loading files from: {path2load}")
    print(f"Solver method: {args.method}")

    try:
        mesh = SWE.FileLoader.load_from_files(path2load)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please check if '{path2load}' exists and contains Config.swe and data files.")
        return

    #Generates the bathymetry paraview file
    SWE.Utilities.render_data(mesh, "bathymetry")

    #Runs the simulation
    SWE.run(mesh, args.method)

if __name__ == "__main__":
    main()
