#!/usr/bin/python3
# -*- coding: Utf-8 -*-
import os
import sys
import atexit
import cupy as np
from datetime import date

import cuSolver as Solver
import cuUtilities as Utilities
import cuFileSaver as FileSaver
import cuFileLoader as FileLoader
import cuBoundaryConditions as BoundaryConditions
import cuAnalyticSolutions as AnalyticSolutions

@atexit.register
def ExitProgram():
    """
    Prints in terminal the program has ended when it is exited

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    print('\n \x1B[32mSHALLOW WATER PROGRAM ENDED!\x1B[0m')

def print_header():
    """
    Prints SWE header showing author information, contact information, and Copyright

    Parameters
    ----------
        None

    Returns
    -------
        None
    """
    os.system('cls' if os.name == 'nt' else 'clear')
    today = date.today()
    today = today.strftime("%m/%d/%Y")

    print( " \033[1;33m                   SHALLOW WATER EQUATIONS       \033[1;0m" )
    print( "    \"A Well-balanced positivity preserving central-upwind scheme    " )
    print( "           on triangular grids for the saint-venant system\"         " )
    print( "                   Departamento de Obras Civiles                     " )
    print( "              Universidad Técnica Federico Santa María               " )
    print( "                         ", today, "                                 " )
    print( "                                                                     " )
    print( " \033[1;34mWritten by:                                      \033[1;0m" )
    print( "   Danilo S. Kusanovic (danilo.kusanov@usm.cl)                       " )
    print( "   Joaquin T. Meza     (joaquin.meza@usm.cl)                         " )
    print( "   Patricio A. Catalán (patricio.catalan@usm.cl)                     " )
    print( "   Juan A. Fuenzalida  (juan.fuenzalidaa@sansano.usm.cl)             " )
    print( "                                                                     " )

    return

def choose_timestep(ts_option,mesh):
    if ts_option.upper() == "RK3":
        from cuSolver import runge_kutta3 as timestep
    elif ts_option.upper() == "RK4":
        from cuSolver import runge_kutta4 as timestep
    elif ts_option.upper() == "FE":
        from cuSolver import forward_euler as timestep
    elif ts_option.upper() == "CRK3":
        from cuSolver import constant_runge_kutta3 as timestep
    elif ts_option.upper() == "CFE":
        from cuSolver import constant_forward_euler as timestep
    elif ts_option.upper() == "FECAS":
        from cuSolver import forward_euler_castro as timestep
        cm=0.2                                                     #Constante mentirosa de Castro
    elif ts_option.upper() == "RK3CAS":
        from cuSolver import runge_kutta3_castro as timestep
        cm=0.2                                                      #Constante mentirosa de Castro
    elif ts_option.upper() == "FECAS2":
        from cuSolver import forward_euler_castro2 as timestep
        cm=0.2                                                     #Constante mentirosa de Castro
    elif ts_option.upper() == "RK3CAS2":
        from cuSolver import runge_kutta3_castro2 as timestep
        cm=0.2                                                      #Constante mentirosa de Castro
    elif ts_option.upper() == "FE3":
        from cuSolver import forward_euler_quad as timestep
        print(" | Preprocessing mesh for quadratic minmod reconstruction... | ")
        Utilities.precomp_minmod2(mesh)
        print(" | Done! | ")
    elif ts_option.upper() == "FEWENO":
        from cuSolver import forward_euler_weno as timestep
        print(" | Preprocessing mesh for quadratic WENO reconstruction... | ")
        Utilities.precomp_weno2(mesh)
        print(" | Done! | ")
    elif ts_option.upper() == "RK3WENO":
        from cuSolver import runge_kutta3_weno as timestep
        print(" | Preprocessing mesh for quadratic WENO reconstruction... | ")
        Utilities.precomp_weno2(mesh)
        print(" | Done! | ")
    elif ts_option.upper() == "RK4WENO":
        from cuSolver import runge_kutta4_weno as timestep
        print(" | Preprocessing mesh for quadratic WENO reconstruction... | ")
        Utilities.precomp_weno2(mesh)
        print(" | Done! | ")
    else:
        print("Timestep option not valid.")
        print("Please choose from:")
        print(">RK3 : 4 step SSP RK3 with linear reconstruction")
        print(">FE  : Forward Euler with linear reconstruction")
        print(">CRK3: 4 step SSP RK3 with constant reconstruction")
        print(">CFE : Forward Euler with constant reconstruction")
        return True

    return timestep

def run(mesh, ts_option, forced_dt=0, movie=True):
    """
    This function runs the simulation after all mesh field have been provided

    Parameters
    ----------
        mesh : Mesh dictionary with the gollowing main fields
            "xj"  : Coordinate x of the midpoint (mass center) of each triangle,[1, nElem]
            "yj"  : Coordinate y of the midpoint (mass center) of each triangle,[1, nElem]
            "nx"  : Coordinate x of the external normal to the element, on each of their sides,[3, nElem]
            "ny"  : Coordinate y of the external normal to the element, on each of their sides,[3, nElem]
            "ljk" : Length of each side of the triangles, [3, nElem]
            "Tj"  : Area of the triangles, [1, nElem]
            "rjk" : Altitude of each side of the triangles, [3, nElem]
            "Wj"  : Water level values located at the mass-center, [1, nElem]
            "HUj" : Water flux x-direction located at the mass-center, [1, nElem]
            "HVj" : Water flux y-direction located at the mass-center, [1, nElem]
            "Bj"  : Bathymetry values located at the mass-center, [1, nElem]
            "Bjk" : Bathymetry values located at the triangle nodes, [3, nElem]
            "Hjk" : Water depth values located at the triangle nodes, [3, nElem]
            "Bmj" : Bathymety values evaluated at side's mid-point, [3, nElem]
            "jk"  : Global indeces of the side's mid-point neighbour element, [1, 3*nElem]
            "Coordinates" : Triangular mesh coordinates pairs (x,y) of each node, [2, nNodes]
            "Elements"    : Triangular mesh connectivity array (i,j,k) of each element, [3, nElem]
            "Neighboors"  : Connectivity array of adjacent elements (j1,j2,j3) for each element in mesh, [3, nElem]
            "GhostCells"  : Ghost element information that emulates soft boundaries, [3, nElem]
            "FilePath"    : Dictionary containing path where such information will be loaded
            "Constants"   : Dictionary containing user defined values
            "InitialConditions" : Dictionary containing path where such information will be loaded
        ts_option: Timestep integrator method, can be
            >RK3 : 4 step SSP RK3 with linear reconstruction
            >FE  : Forward Euler with linear reconstruction
            >CRK3: 4 step SSP RK3 with constant reconstruction
            >CFE : Forward Euler with constant reconstruction
        forced_dt: Desired time step for whole simulation. If left zero, uses variable timestep.

    Returns
    -------
        None 
    """
    #Apply boundary conditions to the mesh and cutoff dry values
    BoundaryConditions.impose(mesh)
    #Solver.cutoff_values(mesh)

    #Save initial conditions and bathymetry for animation
    if movie:
        FileSaver.save_bathymetry(mesh)
        FileSaver.save_animation(mesh, 0)

    #Starts the simulation on the provided mesh
    t  = 0.0
    i  = 0
    dt = 1
    dp = 0
    prompt = True
    cm=0

    #Chooses timestep method
    timestep = choose_timestep(ts_option,mesh)

    #Runs loop
    while (t < mesh["Constants"]["Tmax"] and mesh["Constants"]["Tolerance"] < dt):
        #Solves for this time step
        _,dt,_,_,_ = timestep(mesh,t,1,forced_dt,cm)
        Solver.cutoff_values(mesh)

        dp = dp+100*dt/mesh["Constants"]["Tmax"]

        #Save to disk the solution so far
        t += dt
        i += 1

        #Check if possible divergence
        if i%100 == 0 and not prompt:
            if dp<0.1:
                print("\n\n [WARNING] 100 iterations passed and significant progress hasn't been made.")
                if input("Continue? (y/n): ")[0].upper() not in "Y":
                    FileSaver.save_animation(mesh,t,True)
                    return
                prompt=False
            dp=0

        if movie:
            FileSaver.save_animation(mesh, t)

        print((" | GPU | Running: "+mesh["FilePath"]["Directory"]+" | Progress : %1.2f%s ( t = %1.5f, dt = %1.5f, it = %1d )" % (100*t/mesh["Constants"]["Tmax"], "%", t, dt, i)), flush=True, end="\r")

    #Detects divergence
    if dt <= mesh["Constants"]["Tolerance"] or np.isnan(dt):
        print("\n\n  #################################################")
        print("  #  Divergence detected, check mesh and config!  #")
        print("  #################################################\n")
        FileSaver.save_animation(mesh,t,True)
        raise Exception("DEBUG")
    
    print("\n Avg. dt = %1f\n" % (t/i))

    FileSaver.save_last_Wj(mesh)

    return

def run_error_calc(mesh, ts_option, forced_dt=0, anal_sol=None, params=[]):
    """
    This function runs the simulation after all mesh fields have been provided and calculates the error at each timestep against an analytic solution.

    Parameters
    ----------
        mesh : Mesh dictionary with the gollowing main fields
          "xj"  : Coordinate x of the midpoint (mass center) of each triangle,[1, nElem]
          "yj"  : Coordinate y of the midpoint (mass center) of each triangle,[1, nElem]
          "nx"  : Coordinate x of the external normal to the element, on each of their sides,[3, nElem]
          "ny"  : Coordinate y of the external normal to the element, on each of their sides,[3, nElem]
          "ljk" : Length of each side of the triangles, [3, nElem]
          "Tj"  : Area of the triangles, [1, nElem]
          "rjk" : Altitude of each side of the triangles, [3, nElem]
          "Wj"  : Water level values located at the mass-center, [1, nElem]
          "HUj" : Water flux x-direction located at the mass-center, [1, nElem]
          "HVj" : Water flux y-direction located at the mass-center, [1, nElem]
          "Bj"  : Bathymetry values located at the mass-center, [1, nElem]
          "Bjk" : Bathymetry values located at the triangle nodes, [3, nElem]
          "Hjk" : Water depth values located at the triangle nodes, [3, nElem]
          "Bmj" : Bathymety values evaluated at side's mid-point, [3, nElem]
          "jk"  : Global indeces of the side's mid-point neighbour element, [1, 3*nElem]
          "Coordinates" : Triangular mesh coordinates pairs (x,y) of each node, [2, nNodes]
          "Elements"    : Triangular mesh connectivity array (i,j,k) of each element, [3, nElem]
          "Neighboors"  : Connectivity array of adjacent elements (j1,j2,j3) for each element in mesh, [3, nElem]
          "GhostCells"  : Ghost element information that emulates soft boundaries, [3, nElem]
          "FilePath"    : Dictionary containing path where such information will be loaded
          "Constants"   : Dictionary containing user defined values
          "InitialConditions" : Dictionary containing path where such information will be loaded
        ts_option: Timestep integrator method, can be
            >RK3 : 4 step SSP RK3 with linear reconstruction
            >FE  : Forward Euler with linear reconstruction
            >CRK3: 4 step SSP RK3 with constant reconstruction
            >CFE : Forward Euler with constant reconstruction
        forced_dt: Desired time step for whole simulation. If left zero, uses variable timestep.
        anal_sol : Desired analytical solution to compare against
        params   : List of parameters for that solution

    Returns
    -------
        None 
    """
    #Apply boundary conditions to the mesh 
    BoundaryConditions.impose(mesh)

    #Starts the simulation on the provided mesh
    t  = 0.0
    i  = 0
    dt = 1
    cm=0

    #Chooses timestep method
    if ts_option == "RK3":
        from cuSolver import runge_kutta3 as timestep
    elif ts_option == "FE":
        from cuSolver import forward_euler as timestep
    elif ts_option == "CRK3":
        from cuSolver import constant_runge_kutta3 as timestep
    elif ts_option == "CFE":
        from cuSolver import constant_forward_euler as timestep
    elif ts_option == "FECas":
        cm=0.2
        from cuSolver import forward_euler_castro as timestep
    elif ts_option.upper() == "RK3CAS":
        from cuSolver import runge_kutta3_castro as timestep
        cm=0.2
    elif ts_option.upper() == "FECAS2":
        from cuSolver import forward_euler_castro2 as timestep
        cm=0.2                                                     #Constante mentirosa de Castro
    elif ts_option.upper() == "RK3CAS2":
        from cuSolver import runge_kutta3_castro2 as timestep
        cm=0.2                                                      #Constante mentirosa de Castro
    elif ts_option.upper() == "FE3":
        from cuSolver import forward_euler_quad as timestep
        print(" | Preprocessing mesh for quadratic reconstruction... | ")
        Utilities.precomp_minmod2(mesh)
        print(" | Done! | ")
    else:
        print("Timestep option not valid.")
        print("Please choose from:")
        print(">RK3 : 4 step SSP RK3 with linear reconstruction")
        print(">FE  : Forward Euler with linear reconstruction")
        print(">CRK3: 4 step SSP RK3 with constant reconstruction")
        print(">CFE : Forward Euler with constant reconstruction")
        return

    #Chooses anal. sol.

    if anal_sol == "SWP":
        from cuAnalyticSolutions import solitary_wave_j as anal
    elif anal_sol == "WDB":
        from cuAnalyticSolutions import wet_dam_break_j as anal
        from scipy.optimize import bisect
        cl = np.sqrt(mesh["Constants"]["Gravity"]*params[0])
        cr = np.sqrt(mesh["Constants"]["Gravity"]*params[1])
    
        pol = lambda x: x**6-8*cl**2*x**2*cr**2+16*cl*x**3*cr**2-9*x**4*cr**2-x**2*cr**4+cr**6

        cm = bisect(pol,cl,cr)

        params.append(cm)
    else:
        print("Analytic solution option not valid.")
        print("Please choose from:")
        print(">SWP : Solitary wave propagation")
        print(">WDB : Wet-wet dam break")
        return

    
    AnalyticSolutions.calculate_error(mesh, anal, params, t)

    #Runs loop
    while (t < mesh["Constants"]["Tmax"] and mesh["Constants"]["Tolerance"] < dt):
        #Solves for this time step
        _,dt,_,_,_ = timestep(mesh,t,1,forced_dt,cm)
        Solver.cutoff_values(mesh)
        BoundaryConditions.impose(mesh)

        #Save to disk the solution so far
        t += dt
        i += 1
        AnalyticSolutions.calculate_error(mesh, anal, params, t)

        print((" | GPU | Running: "+mesh["FilePath"]["Directory"]+" (L2 Error) | Progress : %1.2f%s ( t = %1.5f, dt = %1.5f, it = %1d )" % (100*t/mesh["Constants"]["Tmax"], "%", t, dt, i)), flush=True, end="\r")

    #Detects divergence
    if dt <= mesh["Constants"]["Tolerance"]:
        print("#################################################")
        print("#  Divergence detected, check mesh and config!  #")
        print("#################################################\n")
        raise Exception("DEBUG")
    
    print("\n Avg. dt = %1f\n" % (t/i))

    
    FileSaver.save_error_calc(mesh)


    return


def run_with_TS(mesh, ts_option, gauges, points=[], forced_dt=0.):
    """
    This function runs the simulation after all mesh field have been provided and saves the water level at desired list of gauges with timestep mesh["dt_save"]

    Parameters
    ----------
        mesh : Mesh dictionary with the gollowing main fields
          "xj"  : Coordinate x of the midpoint (mass center) of each triangle,[1, nElem]
          "yj"  : Coordinate y of the midpoint (mass center) of each triangle,[1, nElem]
          "nx"  : Coordinate x of the external normal to the element, on each of their sides,[3, nElem]
          "ny"  : Coordinate y of the external normal to the element, on each of their sides,[3, nElem]
          "ljk" : Length of each side of the triangles, [3, nElem]
          "Tj"  : Area of the triangles, [1, nElem]
          "rjk" : Altitude of each side of the triangles, [3, nElem]
          "Wj"  : Water level values located at the mass-center, [1, nElem]
          "HUj" : Water flux x-direction located at the mass-center, [1, nElem]
          "HVj" : Water flux y-direction located at the mass-center, [1, nElem]
          "Bj"  : Bathymetry values located at the mass-center, [1, nElem]
          "Bjk" : Bathymetry values located at the triangle nodes, [3, nElem]
          "Hjk" : Water depth values located at the triangle nodes, [3, nElem]
          "Bmj" : Bathymety values evaluated at side's mid-point, [3, nElem]
          "jk"  : Global indeces of the side's mid-point neighbour element, [1, 3*nElem]
          "Coordinates" : Triangular mesh coordinates pairs (x,y) of each node, [2, nNodes]
          "Elements"    : Triangular mesh connectivity array (i,j,k) of each element, [3, nElem]
          "Neighboors"  : Connectivity array of adjacent elements (j1,j2,j3) for each element in mesh, [3, nElem]
          "GhostCells"  : Ghost element information that emulates soft boundaries, [3, nElem]
          "FilePath"    : Dictionary containing path where such information will be loaded
          "Constants"   : Dictionary containing user defined values
          "InitialConditions" : Dictionary containing path where such information will be loaded
        ts_option: Timestep integrator method, can be
            >RK3 : 4 step SSP RK3 with linear reconstruction
            >FE  : Forward Euler with linear reconstruction
            >CRK3: 4 step SSP RK3 with constant reconstruction
            >CFE : Forward Euler with constant reconstruction
        gauges   : List of element indices to track
        points   : List of tuples with coordinates of points to track (will find elements containing each and add them to gauges)
        forced_dt: Desired time step for whole simulation. If left zero, uses variable timestep.

    Returns
    -------
        None 
    """
    #Apply boundary conditions to the mesh 
    BoundaryConditions.impose(mesh)
    Solver.cutoff_values(mesh)

    #Starts the simulation on the provided mesh
    t  = 0.0
    i  = 0
    dt = 1
    cm=1

    #Chooses timestep method
    timestep = choose_timestep(ts_option,mesh)

    if timestep is bool:
        return
    
    #Add to gauges elements containing points
    if len(points)>0:
        print('Finding gauges...')
        for point in points:
            dists=np.sqrt(np.sum((mesh["Coordinates"].T-np.array(point))**2,axis=1))
            nearest_node=np.argmin(dists)
            elements_with_node=mesh["Elements"][:,np.where(mesh["Elements"]==nearest_node)[1]]
            other_nodes=np.unique(elements_with_node.flatten())
            dists_2=np.sqrt(np.sum((mesh["Coordinates"][:,other_nodes].T-np.array(point))**2,axis=1))
            nearest_node2=other_nodes[np.argpartition(dists_2,2)[1]]
            elements_with_both_nodes=elements_with_node[:,np.where(elements_with_node==nearest_node2)[1]]
            other_nodes2=other_nodes=np.unique(elements_with_both_nodes.flatten())
            dists_3=np.sqrt(np.sum((mesh["Coordinates"][:,other_nodes2].T-np.array(point))**2,axis=1))
            nearest_node3=other_nodes2[np.argpartition(dists_3,3)[2]]
            gauges.append(int(np.where(np.all(np.isin(mesh["Elements"].T,np.array((nearest_node,nearest_node2,nearest_node3))),axis=1))[0][0]))

        gauges=list(set(gauges))

    #Removes files if already exist
    for gauge in gauges:
            filename="gauge_"+str(gauge)+".txt"
            filename = os.path.join(mesh["FilePath"]["Directory"], filename)
            if os.path.exists(filename):
                os.remove(filename)

    #Saves initial state
    FileSaver.save_TS(mesh, t, gauges)

    print(" | Tracking gauges: "+', '.join(str(x) for x in gauges)+" |")

    #Runs loop
    while (t < mesh["Constants"]["Tmax"] and mesh["Constants"]["Tolerance"] < dt):
        #Solves for this time step
        _,dt,_,_,_ = timestep(mesh,t,1,forced_dt,cm)
        Solver.cutoff_values(mesh)

        #Save to disk the solution so far
        i += 1
        t += dt
        
        FileSaver.save_TS(mesh, t, gauges)

        print((" | GPU | Running: "+mesh["FilePath"]["Directory"]+" | Progress : %1.2f%s ( t = %1.5f, dt = %1.5f, it = %1d )" % (100*t/mesh["Constants"]["Tmax"], "%", t, dt, i)), flush=True, end="\r")

    #Detects divergence
    if dt <= mesh["Constants"]["Tolerance"]:
        print("\n  #################################################")
        print("  #  Divergence detected, check mesh and config!  #")
        print("  #################################################\n")
        raise Exception("DEBUG")
    
    print("\n Avg. dt = %1f\n" % (t/i))

    FileSaver.save_animation(mesh,t,True)

    return
    
#Functions to be run when SWE is imported
print_header()
