import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import cuShallowWater as SWE

#Sample script for running multiple cases without using CLI.

print("Running ref mesh")

#Path to location of base mesh
path2load = os.path.join("/home/juan-andres/My Repos/SWEpy-tests/kurganov_sine/", '') # ensure trailing slash
mesh = SWE.FileLoader.load_from_files(path2load)

SWE.run(mesh, "feweno",0,False)

#Loop through submesh refinements
for i in range(1,5):
    print(f"Running mesh sub {i}")
    path2load = os.path.join(f"/home/juan-andres/My Repos/SWEpy-tests/kurganov_sine_sub{i}/", '') # ensure trailing slash
    mesh = SWE.FileLoader.load_from_files(path2load)
    SWE.run(mesh, "feweno",0,False)
    

