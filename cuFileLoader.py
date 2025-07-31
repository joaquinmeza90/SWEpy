import os
import shlex
import cupy as np

import cuUtilities as Utilities

def compute_element_properties(mesh):
    """
    Adds the triangle geometric properties to mesh

    Parameters
    ----------
        mesh : Mesh dictionary to be updated

    Returns
    -------
        No output, but the following keys-value pairs are added to the mesh
          "xj"  : Coordinate x of the midpoint (mass center) of each triangle,[1, nElem]
          "yj"  : Coordinate y of the midpoint (mass center) of each triangle,[1, nElem]
          "nx"  : Coordinate x of the external normal to the element, on each of their sides,[3, nElem]
          "ny"  : Coordinate y of the external normal to the element, on each of their sides,[3, nElem]
          "Tj"  : Area of the triangles, [1, nElem]
          "ljk" : Length of each side of the triangles, [3, nElem]
          "rjk" : Altitude of each side of the triangles, [3, nElem]
    """
    #Unpack elements and coordinates
    Elements = mesh["Elements"]
    Coordinates = mesh["Coordinates"]

    #Compute triangle centroid
    xj = Coordinates[0,Elements].mean(axis=0)
    yj = Coordinates[1,Elements].mean(axis=0)

    #Compute triangular element properties
    nx, ny = Utilities.compute_element_normals(Elements, Coordinates)
    Tj  = Utilities.compute_element_areas(Elements, Coordinates)
    ljk = Utilities.compute_element_edge_lengths(Elements, Coordinates)
    rjk = Utilities.compute_element_edge_altitudes(Elements, Coordinates)

    #Assign variables to dictionary
    mesh["xj" ] = xj
    mesh["yj" ] = yj
    mesh["nx" ] = nx
    mesh["ny" ] = ny
    mesh["ljk"] = ljk
    mesh["Tj" ] = Tj
    mesh["rjk"] = rjk
    
    return

def load_initial_conditions(mesh):
    """
    Adds the Initial Condition data to the mesh, reading from folder in the provided path

    Parameters
    ----------
        mesh : Mesh dictionary to be updated

    Returns
    -------
        No output, but the following keys-value pairs are added to the mesh
          "Wj"  : Water Surface at the midpoints of each element, [1, nElem]
          "HUj" : Discharge on X at the midpoints of each element, [1, nElem]
          "HVj" : Discharge on Y at the midpoints of each element, [1, nElem]
    """
    #Load file information
    path =  mesh["FilePath"]["Directory"]
    Discharge = np.loadtxt(os.path.join(path, mesh["InitialConditions"]["Discharge"]), skiprows=1, dtype=float).T
    WaterLevel = np.loadtxt(os.path.join(path, mesh["InitialConditions"]["WaterLevel"]), skiprows=1, dtype=float).T

    #Gets the values at each node
    Wjk  = WaterLevel[mesh["Elements"]]
    HUjk = Discharge[0,mesh["Elements"]]
    HVjk = Discharge[1,mesh["Elements"]]
    
    #Assign initial condition to state variables
    mesh["Wjk"] = Wjk
    Wj = Wjk.mean(axis=0)
    Wj = np.maximum(Wj, mesh['Bj']) #REVISAR
    Wjk= np.maximum(Wjk,mesh["Bjk"])
    mesh["Wj" ] = Wj 
    mesh["HUj"] = HUjk.mean(axis=0)
    mesh["HVj"] = HVjk.mean(axis=0)

    return

def load_bathymetry_file(mesh,option):
    """
    Adds the bathymetry data to the mesh, reading from folder InitialConditions in provided path

    Parameters
    ----------
        mesh : Mesh dictionary to be updated

    Returns
    -------
        No output, but the following keys-value pairs are added to the mesh
          "Bj"  : Bathymetry values at the midpoints of each element, [1, nElem]
          "Bmj" : Bathymetry values at mid-side of each element, [3, nElem]
          "Bjk" : Bathymetry values at the nodes of each element, [3, nElem]
    """
    #Load file information and computes mid-side and cell center values
    path = mesh["FilePath"]["Directory"]
    Bathymetry = np.loadtxt(os.path.join(path, mesh["FilePath"]["Bathymetry"]), skiprows=1, dtype=float).T

    #Unpack
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]
    xj = Coordinates[0,Elements].mean(axis=0)
    yj = Coordinates[1,Elements].mean(axis=0)
    jk=mesh["jk"]

    #Gets the values at each node
    Bjk = Bathymetry[mesh["Elements"]]
    Bj  = Bjk.mean(axis=0) 
    Bmj, Bx, By = Utilities.bathymetry_mid_point(Bj, Bjk, xj, yj, xjk, yjk, jk, mesh["Neighboors"],mesh["Constants"]["Dry"],option)

    #Assign bathymetry to state variables
    mesh["Bj"] = Bj
    mesh["Bmj"] = Bmj
    mesh["Bjk"] = Bjk 
    mesh["Bx"] = Bx
    mesh["By"] = By
    mesh["Bpoints"] = Bathymetry

    return

def load_configuration_file(mesh):
    """
    This function reads the configuration file (.swe) and extract the data

    Parameters
    ----------
        mesh : Mesh dictionary to be updated

    Returns
    -------
        No output, but the following dictionaries are added to the mesh
          "Constants" : Set of user-defined values
          "FilePath"  : Set of user-defined path where files will be loaded
          "InitialConditions" : Set of user-defined path where initial condition files are located
    """
    #Sets the path to the configuration file
    filepath = os.path.join(mesh["FilePath"]["Directory"], mesh["FilePath"]["Configuration"])

    #Opens file and parse information
    with open(filepath, "r") as fh:
        for line in fh.readlines():
            key, val = list(shlex.split(line))
            
            if key in mesh["Constants"]:
                mesh["Constants"][key] = float(val)
            elif key in mesh["FilePath"]:
                mesh["FilePath"][key] = str(val)
            elif key in mesh["InitialConditions"]:
                mesh["InitialConditions"][key] = str(val)
            else:
                info = Utilities.debug_info(3)
                filename = mesh["FilePath"]["Configuration"]
                print('\x1B[33m ALERT\x1B[0m[line=%d]: The Keyword \'%s\' in \'%s\' file is not recognized!'%(info.lineno,key,filename))
    return

def load_mesh_from_file(mesh):
    """
    This function reads the raw mesh data provided by the user

    Parameters
    ----------
        mesh : Mesh dictionary to be updated

    Returns
    -------
        No output, but the following dictionaries are added to the mesh
          "Coordinates" : Triangular mesh coordinates pairs (x,y) of each node, [2, nNodes]
          "Elements"    : Triangular mesh connectivity array (i,j,k) of each element, [3, nElem]
          "Neighboors"  : Connectivity array of adjacent elements (j1,j2,j3) for each element in mesh, [3, nElem]
          "GhostCells"  : Ghost element information that emulates soft boundaries, [3, nElem]
          "NeighSides"  : Triangular mesh side conectivity array (i,j,k) of each element [3, nElem]
    """
    path =  mesh["FilePath"]["Directory"]
    mesh["NeighSides"] = np.loadtxt(os.path.join(path, mesh["FilePath"]["Sides"]), skiprows=1, dtype=int).T
    Neighboors = np.loadtxt(os.path.join(path, mesh["FilePath"]["Neighboors"]), skiprows=1, dtype=int).T

    #Load mesh information
    mesh["Elements"] = np.loadtxt(os.path.join(path, mesh["FilePath"]["Elements"]), skiprows=1, dtype=int).T
    mesh["GhostCells"  ] = np.loadtxt(os.path.join(path, mesh["FilePath"]["GhostCells"]), skiprows=1, dtype=int).T
    mesh["Coordinates" ] = np.loadtxt(os.path.join(path, mesh["FilePath"]["Coordinates"]), skiprows=1, dtype=float).T

    #Compute side neighbour indeces
    mesh["Neighboors"] = Neighboors
    mesh["jk"] = Utilities.global_index_selection(Neighboors, mesh["NeighSides"]) 

    return

def load_from_files(path2files,option="lbm"):
    """
    Provides with an updated mesh dictionary whose information are loaded from user-defined input files

    Parameters
    ----------
        path2files : Path where the configuration file (.swe) will be loaded

    Returns
    -------
        mesh : Mesh dictionary with fields required for simulation
    """
    #Empty mesh dictionary to store variables
    mesh = Utilities.create_empty_mesh()

    #Gets the configuration file to run simulation
    mesh["FilePath"]["Directory"] = path2files
    for filename in os.listdir(path2files):
        if filename.endswith(".swe"):
            mesh["FilePath"]["Configuration"] = filename
    
    #Configuration file is required
    if not mesh["FilePath"]["Configuration"]:
        info = Utilities.debug_info(2)
        print(" \033[91mERROR\033[1;0m[line=%d]: The configuration file (.swe) must be provided in the path!"%info.lineno)
        exit(-1)

    #Creates the Paraview folder
    paraviewpath = os.path.join(mesh["FilePath"]["Directory"], "Paraview")
    if not os.path.exists(paraviewpath):
        os.makedirs(paraviewpath)

    #Fills out domain information
    load_configuration_file(mesh)
    load_mesh_from_file(mesh)
    load_bathymetry_file(mesh,option)
    load_initial_conditions(mesh)

    #Compute geometric properties for each element
    compute_element_properties(mesh)

    print("Loaded successfully %E nodes and %E cells.\n" % (mesh["Coordinates"].shape[1], mesh["Elements"].shape[1]))

    return mesh
