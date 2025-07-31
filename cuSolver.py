import cupy as np
import cuCentralUpwindMethod as CentralUpwindMethod
import cuPieceWiseReconstruction as PieceWiseReconstruction
import cuBoundaryConditions as BoundaryConditions

def max_time_step(a_in, a_out, rjk, CFL):
    """
    Compute the maximum allowed time step

    Parameters
    ----------
        a_in : One-sided inwards velocity, [3, nElem]
        a_in : One-sided outwards velocity, [3, nElem]
        rjk  : Altitudes at mid-points for each element, [3, nElem]

    Returns
    -------
        dt : Maximum time step, [1,1]
    """
    #Computes Courant-Friedrichs-Lewy for each cell 
    ajk = np.maximum(a_in, a_out)
    val = np.absolute(rjk/ajk)   

    #Maximum allowed time step:
    dt = CFL*val.min().min()

    return dt

def max_time_step_gl(a_in, a_out, rjk, CFL, ghosts):
    """
    Compute the maximum allowed time step

    Parameters
    ----------
        a_in : One-sided inwards velocity, [3, nElem]
        a_in : One-sided outwards velocity, [3, nElem]
        rjk  : Altitudes at mid-points for each element, [3, nElem]

    Returns
    -------
        dt : Maximum time step, [1,1]
    """

    idx_ghost=ghosts[0,0]

    #Computes Courant-Friedrichs-Lewy for each cell 
    ajk = np.maximum(a_in[:,:idx_ghost], a_out[:,:idx_ghost])
    val = np.absolute(rjk[:,:idx_ghost]/ajk)   

    #Maximum allowed time step:
    dt = CFL*val.min().min()

    return dt    

def cutoff_values(mesh):
    """
    Parameters
    ----------
        Wj  : Cell-centered water level, [1,nElem]
        HUj : Cell-centered Flux over X, [1,nElem]
        HVj : Cell-centered Flux over X, [1,nElem]
        DRY : Tolerance that defines when a cell is dry, [1,1]
        TOL : Tolerance that defines when a number is zeroed-out, [1,1]
    
    Returns
    -------
        Wj  : Updated Cell-centered water level, [1,nElem]
        HUj : Updated Cell-centered Flux over X, [1,nElem]
        HVj : Updated Cell-centered Flux over X, [1,nElem]
    """
    #Unpack Variables:
    Bj  = mesh["Bj"]
    Wj  = mesh["Wj"]
    Bjk = mesh["Bjk"]
    Wjk = mesh["Wjk"]
    HUj = mesh["HUj"]
    HVj = mesh["HVj"]
    DRY = mesh["Constants"]["Dry"]
    TOL = mesh["Constants"]["Tolerance"]

    #Update Water level values
    ind = (Wj - Bj) < DRY
    Wj[ind] = Bj[ind]
    ind = (Wjk - Bjk) < DRY
    Wjk[ind] = Bjk[ind]

    #Update Flux along x-direction values
    ind = np.fabs(HUj) < TOL
    HUj[ind] = 0.000

    #Update Flux along y-direction values
    ind = np.fabs(HVj) < TOL
    HVj[ind] = 0.000

    mesh["Wj"]  = Wj
    mesh["Wjk"] = Wjk
    mesh["HUj"] = HUj
    mesh["HVj"] = HVj

    return

def forward_euler(mesh,t,rkc=1,rkdt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 1 step forward Euler numerical integration

    Parameters
    ----------
        mesh : Mesh dictionary with the following fields:
          "Wj"  : Cell-centered water level, [1,nElem]
          "HUj" : Cell-centered Flux over X, [1,nElem]
          "HVj" : Cell-centered Flux over Y, [1,nElem]
          "ljk" : Side lengths for each triangle, [3,nElem]
          "rjk" : Side altitudes for each triangle, [3,nElem]
          "Tj"  : Area of the j-th Triangle, [1,nElem]
          "nx"  : Unit-Normal component over X, [3,nElem]
          "ny"  : Unit-Normal component over Y, [3,nElem]
          "g"   : Gravity constant, [1,1]
          "n"   : Manning's number, [1,nElem]
          "DRY" : Maximum water level to be consider as zero, [1,1]
        t    : Current simulation time (currently unused)
        rkc  : Runge-Kutta constant for use in RK integration
        rkdt : Timestep to be used. If not provided, function calculates it with CFL condition

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """
    #UNPACK MESH VARIABLES
    Bmj = mesh["Bmj"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    #PIECEWISE RECONSTRUCTION
    wx, wy = PieceWiseReconstruction.minmod(mesh["Wj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    Wmj  = PieceWiseReconstruction.set_linear_midpoint_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    mesh["Wjk"] = PieceWiseReconstruction.set_linear_vertices_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HUj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HUmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HUj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HVj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HVmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HVj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)

    #Well Balanced wet-dry Reconstruction
    Hmj     = CentralUpwindMethod.set_water_depth(Wmj, Bmj, mesh["Constants"]["Dry"])
    mesh["Wjk"],Wmj,Hmj,wx,wy = PieceWiseReconstruction.set_linear_well_balanced_wet_dry(mesh["Wjk"], mesh["Bjk"], Wmj, Hmj, mesh["Bmj"], Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy)
    
    
    #Computation of the desingularizated velocities
    HUmj, umj  = CentralUpwindMethod.midvelocity(HUmj, Hmj, mesh["Constants"]["Dry"])
    HVmj, vmj  = CentralUpwindMethod.midvelocity(HVmj, Hmj, mesh["Constants"]["Dry"])

    #Computes the one-sided propagation velocity
    a_out, a_in = CentralUpwindMethod.one_sided_speed(Hmj, umj, vmj, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    #Computes the maximum time step
    mesh["dt"] = max_time_step(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"]) if rkdt == 0 else rkdt

    #NUMERICAL INTEGRATION TO NEXT TIME STEP
    #Bottom Friction Values
    Fr = CentralUpwindMethod.friction_term(HUmj, HVmj, mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    #Source Term Values
    Sx, Sy = CentralUpwindMethod.source_term(Hmj, wx, wy, mesh["ljk"], mesh["Tj"], mesh["nx"], mesh["ny"], mesh["Constants"]["Gravity"])
    fx, fy = CentralUpwindMethod.coriolis(mesh["HUj"],mesh["HVj"],mesh["Constants"]["coriolis"])

    #Water level surface (Forward Euler)
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.water_level_function(Wmj, HUmj, HVmj, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    mesh["Wj"] = mesh["Wj"]  + (mesh["dt"]*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"])/rkc #Acá iría el RK3

    #Flux along X-direction
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.flux_function_x(Hmj, HUmj, HVmj, a_in, a_out, umj, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HUj"] = mesh["HUj"] + (mesh["dt"]*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]  + mesh["dt"]*Sx + mesh["dt"]*fx)/rkc #acá igual

    #Flux along Y-direction
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.flux_function_y(Hmj, HUmj, HVmj, a_in, a_out, vmj, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HVj"] = mesh["HVj"] + (mesh["dt"]*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]  + mesh["dt"]*Sy + mesh["dt"]*fy)/rkc #acá igual
    
    #Update value using the friction
    mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)
    
    #Computed for ploting purposes
    #mesh["Hj"] = np.maximum(mesh["Wj"] - mesh["Bj"], mesh["Constants"]["Dry"])

    BoundaryConditions.impose(mesh)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def runge_kutta3(mesh,t,c,dt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 4-stage SSP RK3 numerical integration

    Parameters
    ----------
        mesh : Mesh dictionary
        t    : Current simulation time (currently unused)

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """

    ftn=np.array([mesh["Wj"],mesh["HUj"],mesh["HVj"]])                                         #0th step
    dt=forward_euler(mesh,t,2,dt,0)[1]                                                         #1st step, forward euler of last step with 1/2 dt, saves dt
    forward_euler(mesh,t,2,dt,cm)                                                              #2nd step, forward euler of last step with 1/2 dt
    mesh["Wj"],mesh["HUj"],mesh["HVj"]=2*ftn/3+(np.array(forward_euler(mesh,t,2,dt,0)[2:]))/3  #3rd step, forces mesh to be 2/3 0th step + 1/3 forward euler of last step with 1/2 dt
    forward_euler(mesh,t,2,dt,cm)                                                              #4th step, forward euler of last step with 1/2 dt

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def constant_forward_euler(mesh,t,rkc=1,rkdt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 1 step forward Euler numerical integration with constant reconstructions

    Parameters
    ----------
        mesh : Mesh dictionary with the following fields:
          "Wj"  : Cell-centered water level, [1,nElem]
          "HUj" : Cell-centered Flux over X, [1,nElem]
          "HVj" : Cell-centered Flux over Y, [1,nElem]
          "ljk" : Side lengths for each triangle, [3,nElem]
          "rjk" : Side altitudes for each triangle, [3,nElem]
          "Tj"  : Area of the j-th Triangle, [1,nElem]
          "nx"  : Unit-Normal component over X, [3,nElem]
          "ny"  : Unit-Normal component over Y, [3,nElem]
          "g"   : Gravity constant, [1,1]
          "n"   : Manning's number, [1,nElem]
          "DRY" : Maximum water level to be consider as zero, [1,1]
        t    : Current simulation time (currently unused)
        rkc  : Runge-Kutta constant for use in RK integration
        rkdt : Timestep to be used. If not provided, function calculates it with CFL condition

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """
    #UNPACK MESH VARIABLES
    Bmj = mesh["Bmj"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    #PIECEWISE RECONSTRUCTION
    Wmj  = PieceWiseReconstruction.set_constant_midpoint_values(mesh["Wj"])
    HUmj = PieceWiseReconstruction.set_constant_midpoint_values(mesh["HUj"])
    HVmj = PieceWiseReconstruction.set_constant_midpoint_values(mesh["HVj"])

    #Well Balanced Reconstruction
    Hmj     = CentralUpwindMethod.set_water_depth(Wmj, Bmj, mesh["Constants"]["Dry"])
    mesh["Wjk"],Wmj,Hmj = PieceWiseReconstruction.set_constant_well_balanced_wet_dry(mesh["Wjk"], mesh["Bjk"], Wmj, Hmj, mesh["Bmj"], Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"])
    
    #Computation of the desingularizated velocities
    HUmj, umj  = CentralUpwindMethod.midvelocity(HUmj, Hmj, mesh["Constants"]["Dry"])
    HVmj, vmj  = CentralUpwindMethod.midvelocity(HVmj, Hmj, mesh["Constants"]["Dry"])

    #Computes the one-sided propagation velocity
    a_out, a_in = CentralUpwindMethod.one_sided_speed(Hmj, umj, vmj, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    #Computes the maximum time step
    mesh["dt"] = max_time_step(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"]) if not rkdt else rkdt

    #NUMERICAL INTEGRATION TO NEXT TIME STEP
    #Bottom Friction Values
    Fr = CentralUpwindMethod.friction_term(HUmj, HVmj, mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    #Source Term Values
    Sx, Sy = CentralUpwindMethod.source_term(Hmj, 0, 0, mesh["ljk"], mesh["Tj"], mesh["nx"], mesh["ny"], mesh["Constants"]["Gravity"])
    fx, fy = CentralUpwindMethod.coriolis(mesh["HUj"],mesh["HVj"],mesh["Constants"]["coriolis"])

    #Water level surface (Forward Euler)
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.water_level_function(Wmj, HUmj, HVmj, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    mesh["Wj"] = mesh["Wj"]  + (mesh["dt"]*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"])/rkc

    #Flux along X-direction
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.flux_function_x(Hmj, HUmj, HVmj, a_in, a_out, umj, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HUj"] = mesh["HUj"] + (mesh["dt"]*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]  + mesh["dt"]*Sx + mesh["dt"]*fx)/rkc

    #Flux along Y-direction
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.flux_function_y(Hmj, HUmj, HVmj, a_in, a_out, vmj, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HVj"] = mesh["HVj"] + (mesh["dt"]*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]  + mesh["dt"]*Sy + mesh["dt"]*fy)/rkc
    
    #Update value using the friction
    mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)

    #Calculate values at the vertices for plotting
    mesh["Wjk"] = PieceWiseReconstruction.set_constant_vertices_values(mesh["Wj"])
    
    #Computed for ploting purposes
    #mesh["Hj"] = np.maximum(mesh["Wj"] - mesh["Bj"], mesh["Constants"]["Dry"])

    BoundaryConditions.impose(mesh)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def constant_runge_kutta3(mesh,t,c,dt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 4-stage SSP RK3 numerical integration with constant reconstructions

    Parameters
    ----------
        mesh : Mesh dictionary
        t    : Current simulation time (currently unused)

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """

    ftn=np.array([mesh["Wj"],mesh["HUj"],mesh["HVj"]])                                               #0th step
    dt=constant_forward_euler(mesh,t,2,dt,cm)[1]                                                           #1st step, forward euler of last step with 1/2 dt, saves dt
    constant_forward_euler(mesh,t,2,dt,cm)                                                              #2nd step, forward euler of last step with 1/2 dt
    mesh["Wj"],mesh["HUj"],mesh["HVj"]=2*ftn/3+(np.array(constant_forward_euler(mesh,t,2,dt,cm)[2:]))/3 #3rd step, forces mesh to be 2/3 0th step + 1/3 forward euler of last step with 1/2 dt
    constant_forward_euler(mesh,t,2,dt,cm)                                                              #4th step, forward euler of last step with 1/2 dt

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def forward_euler_castro(mesh,t,rkc=1,rkdt=0,cm=0):
    #UNPACK MESH VARIABLES
    Bmj = mesh["Bmj"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    #PIECEWISE RECONSTRUCTION
    wx, wy = PieceWiseReconstruction.minmod(mesh["Wj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    Wmj  = PieceWiseReconstruction.set_linear_midpoint_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    mesh["Wjk"] = PieceWiseReconstruction.set_linear_vertices_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HUj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HUmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HUj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HVj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HVmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HVj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)

    #Well Balanced wet-dry Reconstruction
    Hmj     = CentralUpwindMethod.set_water_depth(Wmj, Bmj, mesh["Constants"]["Dry"])
    mesh["Wjk"],Wmj,Hmj,wx,wy = PieceWiseReconstruction.set_linear_well_balanced_wet_dry(mesh["Wjk"], mesh["Bjk"], Wmj, Hmj, mesh["Bmj"], Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy)
    
    
    #Computation of the desingularizated velocities
    HUmj, umj  = CentralUpwindMethod.midvelocity(HUmj, Hmj, mesh["Constants"]["Dry"])
    HVmj, vmj  = CentralUpwindMethod.midvelocity(HVmj, Hmj, mesh["Constants"]["Dry"])

    #Computes the one-sided propagation velocity
    a_out, a_in = CentralUpwindMethod.one_sided_speed(Hmj, umj, vmj, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    #Computes the maximum time step
    mesh["dt"] = max_time_step(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"]) if not rkdt else rkdt

    Wi=np.array((mesh["Wj"],mesh["HUj"],mesh["HVj"]))

    ch = CentralUpwindMethod.flux_castro(mesh,Hmj,HUmj,HVmj,umj,vmj,xjk,yjk,cm)

    Wi = Wi-(mesh["dt"]/mesh["Tj"])*(ch)/rkc

    mesh["Wj"] = Wi[0,:]
    mesh["HUj"] = Wi[1,:]
    mesh["HVj"] = Wi[2,:]

    BoundaryConditions.impose(mesh)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def runge_kutta3_castro(mesh,t,c,dt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 4-stage SSP RK3 numerical integration with constant reconstructions

    Parameters
    ----------
        mesh : Mesh dictionary
        t    : Current simulation time (currently unused)

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """

    ftn=np.array([mesh["Wj"],mesh["HUj"],mesh["HVj"]])                                               #0th step
    dt=forward_euler_castro(mesh,t,2,dt,cm)[1]                                                           #1st step, forward euler of last step with 1/2 dt, saves dt
    forward_euler_castro(mesh,t,2,dt,cm)                                                              #2nd step, forward euler of last step with 1/2 dt
    mesh["Wj"],mesh["HUj"],mesh["HVj"]=2*ftn/3+(np.array(forward_euler_castro(mesh,t,2,dt,cm)[2:]))/3 #3rd step, forces mesh to be 2/3 0th step + 1/3 forward euler of last step with 1/2 dt
    forward_euler_castro(mesh,t,2,dt,cm)                                                              #4th step, forward euler of last step with 1/2 dt

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def forward_euler_castro2(mesh,t,rkc,rkdt=0,cm=0):
    #UNPACK MESH VARIABLES
    g = mesh["Constants"]["Gravity"]
    Bmj = mesh["Bmj"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    #PIECEWISE RECONSTRUCTION
    wx, wy = PieceWiseReconstruction.minmod(mesh["Wj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    Wmj  = PieceWiseReconstruction.set_linear_midpoint_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    mesh["Wjk"] = PieceWiseReconstruction.set_linear_vertices_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HUj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HUmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HUj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HVj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HVmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HVj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)

    #Well Balanced wet-dry Reconstruction
    Hmj     = CentralUpwindMethod.set_water_depth(Wmj, Bmj, mesh["Constants"]["Dry"])
    mesh["Wjk"],Wmj,Hmj,wx,wy = PieceWiseReconstruction.set_linear_well_balanced_wet_dry(mesh["Wjk"], mesh["Bjk"], Wmj, Hmj, mesh["Bmj"], Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy)
    
    #Computation of the desingularizated velocities
    HUmj, umj  = CentralUpwindMethod.midvelocity(HUmj, Hmj, mesh["Constants"]["Dry"])
    HVmj, vmj  = CentralUpwindMethod.midvelocity(HVmj, Hmj, mesh["Constants"]["Dry"])

    #Bottom Friction Values
    Fr = CentralUpwindMethod.friction_term(HUmj, HVmj, mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    flujo0, dt0 = CentralUpwindMethod.flux_castro_2(mesh,Hmj,HUmj,HVmj,umj,vmj,0,cm)
    flujo1, dt1 = CentralUpwindMethod.flux_castro_2(mesh,Hmj,HUmj,HVmj,umj,vmj,1,cm)
    flujo2, dt2 = CentralUpwindMethod.flux_castro_2(mesh,Hmj,HUmj,HVmj,umj,vmj,2,cm)

    Wi=np.array((mesh["Wj"],mesh["HUj"],mesh["HVj"]))

    mesh["dt"] = np.min(np.array([dt0,dt1,dt2])) if not rkdt else rkdt

    vint=mesh["Tj"]*np.array([np.zeros_like(wx),g*wx*(Hmj[0,:]+Hmj[1,:]+Hmj[2,:])/3,g*wy*(Hmj[0,:]+Hmj[1,:]+Hmj[2,:])/3])

    Wi = Wi - (mesh["dt"]/mesh["Tj"])*(flujo0*mesh["ljk"][0,:]+flujo1*mesh["ljk"][1,:]+flujo2*mesh["ljk"][2,:]+vint)/rkc

    mesh["Wj"] = Wi[0,:]
    mesh["HUj"] = Wi[1,:]
    mesh["HVj"] = Wi[2,:]
    
    #Update value using the friction
    mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)

    BoundaryConditions.impose(mesh)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def runge_kutta3_castro2(mesh,t,c,dt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 4-stage SSP RK3 numerical integration with constant reconstructions

    Parameters
    ----------
        mesh : Mesh dictionary
        t    : Current simulation time (currently unused)

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """

    ftn=np.array([mesh["Wj"],mesh["HUj"],mesh["HVj"]])                                               #0th step
    dt=forward_euler_castro2(mesh,t,2,dt,cm)[1]                                                           #1st step, forward euler of last step with 1/2 dt, saves dt
    forward_euler_castro2(mesh,t,2,dt,cm)                                                              #2nd step, forward euler of last step with 1/2 dt
    mesh["Wj"],mesh["HUj"],mesh["HVj"]=2*ftn/3+(np.array(forward_euler_castro2(mesh,t,2,dt,cm)[2:]))/3 #3rd step, forces mesh to be 2/3 0th step + 1/3 forward euler of last step with 1/2 dt
    forward_euler_castro2(mesh,t,2,dt,cm)                                                              #4th step, forward euler of last step with 1/2 dt

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def forward_euler_quad(mesh,t,rkc=1,rkdt=0,cm=0):

    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 1 step forward Euler numerical integration using quadratic reconstruction at each cell.

    Parameters
    ----------
        mesh : Mesh dictionary with the following fields:
          "Wj"  : Cell-centered water level, [1,nElem]
          "HUj" : Cell-centered Flux over X, [1,nElem]
          "HVj" : Cell-centered Flux over Y, [1,nElem]
          "ljk" : Side lengths for each triangle, [3,nElem]
          "rjk" : Side altitudes for each triangle, [3,nElem]
          "Tj"  : Area of the j-th Triangle, [1,nElem]
          "nx"  : Unit-Normal component over X, [3,nElem]
          "ny"  : Unit-Normal component over Y, [3,nElem]
          "g"   : Gravity constant, [1,1]
          "n"   : Manning's number, [1,nElem]
          "DRY" : Maximum water level to be consider as zero, [1,1]
        t    : Current simulation time (currently unused)
        rkc  : Runge-Kutta constant for use in RK integration
        rkdt : Timestep to be used. If not provided, function calculates it with CFL condition

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """
    #UNPACK MESH VARIABLES
    Bmj1 = mesh["Bmj1"]
    Bmj2 = mesh["Bmj2"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    #PIECEWISE RECONSTRUCTION
    Lx, Ly, Lxx, Lyy, Lxy = PieceWiseReconstruction.minmod2(mesh["Wj"],  mesh["Stencils"], mesh["NChoose"], mesh["rjk"], xjk, yjk, mesh["xj"], mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"], mesh["Constants"]["Tolerance"])
    Wmj1, Wmj2  = PieceWiseReconstruction.set_quadratic_midpoint_values(mesh["Wj"],Lx,Ly,Lxx,Lyy,Lxy,mesh["xj"],mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"],xjk,yjk)
    Wg1, Wg2, Wg3 = PieceWiseReconstruction.set_quadratic_in_values(mesh["Wj"],Lx,Ly,Lxx,Lyy,Lxy,mesh["xj"],mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"],mesh["xg0"], mesh["xg1"], mesh["xg2"], mesh["yg0"], mesh["yg1"], mesh["yg2"])
    wx1,wx2,wx3,wy1,wy2,wy3 = PieceWiseReconstruction.set_quadratic_din_values(Lx,Ly,Lxx,Lyy,Lxy,mesh["xj"],mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"],mesh["xg0"], mesh["xg1"], mesh["xg2"], mesh["yg0"], mesh["yg1"], mesh["yg2"])
    mesh["Wjk"] = PieceWiseReconstruction.set_quadratic_vertices_values(mesh["Wj"],wx,wy,Lxx,Lyy,Lxy,mesh["xj"],mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"],xjk,yjk)
    Lx, Ly, Lxx, Lyy, Lxy = PieceWiseReconstruction.minmod2(mesh["HUj"], mesh["Stencils"], mesh["NChoose"], mesh["rjk"], xjk, yjk, mesh["xj"], mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"], mesh["Constants"]["Tolerance"])
    HUmj1, HUmj2  = PieceWiseReconstruction.set_quadratic_midpoint_values(mesh["HUj"],Lx,Ly,Lxx,Lyy,Lxy,mesh["xj"],mesh["yj"],xjk,yjk)
    Lx, Ly, Lxx, Lyy, Lxy = PieceWiseReconstruction.minmod2(mesh["HVj"], mesh["Stencils"], mesh["NChoose"], mesh["rjk"], xjk, yjk, mesh["xj"], mesh["yj"],mesh["Ix"],mesh["Iy"],mesh["Ixy"],mesh["Tj"], mesh["Constants"]["Tolerance"])
    HVmj1, HVmj2  = PieceWiseReconstruction.set_quadratic_midpoint_values(mesh["HVj"],Lx,Ly,Lxx,Lyy,Lxy,mesh["xj"],mesh["yj"],xjk,yjk)

    #Well Balanced wet-dry Reconstruction
    Hmj1     = CentralUpwindMethod.set_water_depth(Wmj1, Bmj1, mesh["Constants"]["Dry"])
    Hmj2     = CentralUpwindMethod.set_water_depth(Wmj2, Bmj2, mesh["Constants"]["Dry"])
    #TO-DO!!! mesh["Wjk"],Wmj,Hmj,wx,wy = PieceWiseReconstruction.set_linear_well_balanced_wet_dry(mesh["Wjk"], mesh["Bjk"], Wmj, Hmj, mesh["Bmj"], Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy)
    
    
    #Computation of the desingularizated velocities
    HUmj1, umj1  = CentralUpwindMethod.midvelocity(HUmj1, Hmj1, mesh["Constants"]["Dry"])
    HVmj1, vmj1  = CentralUpwindMethod.midvelocity(HVmj1, Hmj1, mesh["Constants"]["Dry"])

    HUmj2, umj2  = CentralUpwindMethod.midvelocity(HUmj2, Hmj2, mesh["Constants"]["Dry"])
    HVmj2, vmj2  = CentralUpwindMethod.midvelocity(HVmj2, Hmj2, mesh["Constants"]["Dry"])

    #Computes the one-sided propagation velocity
    a_out1, a_in1 = CentralUpwindMethod.one_sided_speed(Hmj1, umj1, vmj1, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    a_out2, a_in2 = CentralUpwindMethod.one_sided_speed(Hmj2, umj2, vmj2, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    a_out = np.maximum(a_out1,a_out2)
    a_in = -np.minimum(-a_in1,-a_in2) #Iffy about this

    #Computes the maximum time step
    mesh["dt"] = max_time_step(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"]) if rkdt == 0 else rkdt

    #NUMERICAL INTEGRATION TO NEXT TIME STEP
    #Bottom Friction Values
    Fr = CentralUpwindMethod.friction_term(0.5*(HUmj1+HUmj2), 0.5*(HVmj1+HVmj2), mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    #Source Term Values
    Sx, Sy = CentralUpwindMethod.source_term2(Hmj1, Hmj2, Wg1,Wg2,Wg3, mesh["Bg0"], mesh["Bg1"], mesh["Bg2"], wx1,wx2,wx3, wy1,wy2,wy3, mesh["ljk"], mesh["Tj"], mesh["nx"], mesh["ny"], mesh["Constants"]["Gravity"])

    #Water level surface (Forward Euler)
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.water_level_function(Wmj1, HUmj1, HVmj1, a_in1, a_out1, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.water_level_function(Wmj2, HUmj2, HVmj2, a_in2, a_out2, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    mesh["Wj"] = mesh["Wj"]  + (mesh["dt"]*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"])/rkc #Acá iría el RK3

    #Flux along X-direction
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.flux_function_x(Hmj1, HUmj1, HVmj1, a_in1, a_out1, umj1, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.flux_function_x(Hmj2, HUmj2, HVmj2, a_in2, a_out2, umj2, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HUj"] = mesh["HUj"] + (mesh["dt"]*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"]  + mesh["dt"]*Sx)/rkc #acá igual

    #Flux along Y-direction
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.flux_function_y(Hmj1, HUmj1, HVmj1, a_in1, a_out1, umj1, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.flux_function_y(Hmj2, HUmj2, HVmj2, a_in2, a_out2, umj2, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HVj"] = mesh["HVj"] + (mesh["dt"]*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"]  + mesh["dt"]*Sy)/rkc #acá igual
    
    #Update value using the friction
    mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)
    
    #Computed for ploting purposes
    #mesh["Hj"] = np.maximum(mesh["Wj"] - mesh["Bj"], mesh["Constants"]["Dry"])

    BoundaryConditions.impose(mesh)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def forward_euler_weno(mesh,t,rkc=1,rkdt=0,cm=0):

    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 1 step forward Euler numerical integration using quadratic reconstruction at each cell.

    Parameters
    ----------
        mesh : Mesh dictionary with the following fields:
          "Wj"  : Cell-centered water level, [1,nElem]
          "HUj" : Cell-centered Flux over X, [1,nElem]
          "HVj" : Cell-centered Flux over Y, [1,nElem]
          "ljk" : Side lengths for each triangle, [3,nElem]
          "rjk" : Side altitudes for each triangle, [3,nElem]
          "Tj"  : Area of the j-th Triangle, [1,nElem]
          "nx"  : Unit-Normal component over X, [3,nElem]
          "ny"  : Unit-Normal component over Y, [3,nElem]
          "g"   : Gravity constant, [1,1]
          "n"   : Manning's number, [1,nElem]
          "DRY" : Maximum water level to be consider as zero, [1,1]
        t    : Current simulation time (currently unused)
        rkc  : Runge-Kutta constant for use in RK integration
        rkdt : Timestep to be used. If not provided, function calculates it with CFL condition

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """
    #UNPACK MESH VARIABLES
    Bmj1 = mesh["Bmj1"]
    Bmj2 = mesh["Bmj2"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    ov=mesh["no_2nd_neigh"] | mesh["no_1st_neigh"]
    choice = mesh["cChoose1"]
    
    #PIECEWISE RECONSTRUCTION
    wrecon, wx, wy = PieceWiseReconstruction.weno2(mesh["Wj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])
    Wmj1, Wmj2  = PieceWiseReconstruction.set_weno_midpoint_values(mesh["Wj"],wrecon,xjk,yjk,ov)
    Wg1, Wg2, Wg3 = PieceWiseReconstruction.set_weno_in_values(wrecon, mesh["xg0"], mesh["xg1"], mesh["xg2"], mesh["yg0"], mesh["yg1"], mesh["yg2"],ov)
    mesh["Wjk"] = PieceWiseReconstruction.set_weno_vertices_values(mesh["Wj"],wrecon,xjk,yjk,ov)
    recon = PieceWiseReconstruction.weno2(mesh["HUj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])[0]
    HUmj1, HUmj2  = PieceWiseReconstruction.set_weno_midpoint_values(mesh["HUj"],recon,xjk,yjk,ov)
    recon = PieceWiseReconstruction.weno2(mesh["HVj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])[0]
    HVmj1, HVmj2  = PieceWiseReconstruction.set_weno_midpoint_values(mesh["HVj"],recon,xjk,yjk,ov)

    #Well Balanced wet-dry Reconstruction
    Hmj1     = CentralUpwindMethod.set_water_depth(Wmj1, Bmj1, mesh["Constants"]["Dry"])
    Hmj2     = CentralUpwindMethod.set_water_depth(Wmj2, Bmj2, mesh["Constants"]["Dry"])
    #mesh["Wjk"],Wmj1,Wmj2,Hmj1,Hmj2,wxc,wyc,wreconc = PieceWiseReconstruction.set_linear_well_balanced_wet_dry2(mesh["Wjk"], mesh["Bjk"], Wmj1, Wmj2, Hmj1, Hmj2, Bmj1, Bmj2, Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy,wrecon)
    wxc=wx
    wyc=wy
    #Wg1, Wg2, Wg3 = PieceWiseReconstruction.set_weno_in_values(wreconc, mesh["xg0"], mesh["xg1"], mesh["xg2"], mesh["yg0"], mesh["yg1"], mesh["yg2"],ov)

    #This should update the inside gaussian points for W too!
    
    #Computation of the desingularizated velocities
    HUmj1, umj1  = CentralUpwindMethod.midvelocity(HUmj1, Hmj1, mesh["Constants"]["Dry"])
    HVmj1, vmj1  = CentralUpwindMethod.midvelocity(HVmj1, Hmj1, mesh["Constants"]["Dry"])

    HUmj2, umj2  = CentralUpwindMethod.midvelocity(HUmj2, Hmj2, mesh["Constants"]["Dry"])
    HVmj2, vmj2  = CentralUpwindMethod.midvelocity(HVmj2, Hmj2, mesh["Constants"]["Dry"])

    #Calculate the outer values for each variable
    Wmj1jk = Wmj1.take(mesh["jk"])
    Wmj2jk = Wmj2.take(mesh["jk"])
    Wmj1out = np.where(choice==0, Wmj1jk, Wmj2jk)
    Wmj2out = np.where(choice==1, Wmj1jk, Wmj2jk)

    Hmj1jk = Hmj1.take(mesh["jk"])
    Hmj2jk = Hmj2.take(mesh["jk"])
    Hmj1out = np.where(choice==0, Hmj1jk, Hmj2jk)
    Hmj2out = np.where(choice==1, Hmj1jk, Hmj2jk)

    umj1jk = umj1.take(mesh["jk"])
    umj2jk = umj2.take(mesh["jk"])
    umj1out = np.where(choice==0, umj1jk, umj2jk)
    umj2out = np.where(choice==1, umj1jk, umj2jk)

    vmj1jk = vmj1.take(mesh["jk"])
    vmj2jk = vmj2.take(mesh["jk"])
    vmj1out = np.where(choice==0, vmj1jk, vmj2jk)
    vmj2out = np.where(choice==1, vmj1jk, vmj2jk)

    HUmj1jk = HUmj1.take(mesh["jk"])
    HUmj2jk = HUmj2.take(mesh["jk"])
    HUmj1out = np.where(choice==0, HUmj1jk, HUmj2jk)
    HUmj2out = np.where(choice==1, HUmj1jk, HUmj2jk)

    HVmj1jk = HVmj1.take(mesh["jk"])
    HVmj2jk = HVmj2.take(mesh["jk"])
    HVmj1out = np.where(choice==0, HVmj1jk, HVmj2jk)
    HVmj2out = np.where(choice==1, HVmj1jk, HVmj2jk)

    #Computes the one-sided propagation velocity
    a_out, a_in = CentralUpwindMethod.one_sided_speed2(Hmj1, umj1, vmj1, Hmj2, umj2, vmj2, Hmj1out, umj1out, vmj1out, Hmj2out,umj2out,vmj2out, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    #Computes the maximum time step
    mesh["dt"] = max_time_step_gl(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"],mesh["GhostCells"]) if rkdt == 0 else rkdt

    #NUMERICAL INTEGRATION TO NEXT TIME STEP
    #Bottom Friction Values [ T  O    D  O !!!!!!!! ]
    Fr = CentralUpwindMethod.friction_term(0.5*(HUmj1+HUmj2), 0.5*(HVmj1+HVmj2), mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    #Source Term Values
    Sx, Sy = CentralUpwindMethod.source_term2(Hmj1, Hmj2, Wg1, Wg2, Wg3, mesh["Bg0"], mesh["Bg1"], mesh["Bg2"], wxc(mesh["xg0"],mesh["yg0"],ov), wxc(mesh["xg1"],mesh["yg1"],ov), wxc(mesh["xg2"],mesh["yg2"],ov), wyc(mesh["xg0"],mesh["yg0"],ov), wyc(mesh["xg1"],mesh["yg1"],ov), wyc(mesh["xg2"],mesh["yg2"],ov), mesh["ljk"], mesh["Tj"], mesh["nx"], mesh["ny"], mesh["Constants"]["Gravity"])
    fx, fy = CentralUpwindMethod.coriolis(mesh["HUj"],mesh["HVj"],mesh["Constants"]["coriolis"])

    #Water level surface (Forward Euler)
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.water_level_function2(Wmj1, HUmj1, HVmj1, Wmj1out, HUmj1out, HVmj1out, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.water_level_function2(Wmj2, HUmj2, HVmj2, Wmj2out, HUmj2out, HVmj2out, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    mesh["Wj"] = mesh["Wj"]   + (mesh["dt"]*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"])/rkc #Acá iría el RK3

    #Flux along X-direction
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.flux_function_x2(Hmj1, HUmj1, HVmj1, Hmj1out, HUmj1out, HVmj1out, a_in, a_out, umj1, umj1out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.flux_function_x2(Hmj2, HUmj2, HVmj2, Hmj2out, HUmj2out, HVmj2out, a_in, a_out, umj2, umj2out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HUj"] = mesh["HUj"] + (mesh["dt"]*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"] + mesh["dt"]*Sx + mesh["dt"]*fx)/rkc #acá igual

    #Flux along Y-direction
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.flux_function_y2(Hmj1, HUmj1, HVmj1, Hmj1out, HUmj1out, HVmj1out, a_in, a_out, vmj1, vmj1out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.flux_function_y2(Hmj2, HUmj2, HVmj2, Hmj2out, HUmj2out, HVmj2out, a_in, a_out, vmj2, vmj2out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    mesh["HVj"] = mesh["HVj"] + (mesh["dt"]*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"] + mesh["dt"]*Sy + mesh["dt"]*fy)/rkc #acá igual
    
    #Update value using the friction
    #mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    #mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)
    
    #Computed for ploting purposes
    #mesh["Hj"] = np.maximum(mesh["Wj"] - mesh["Bj"], mesh["Constants"]["Dry"])

    BoundaryConditions.impose(mesh)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def runge_kutta3_weno(mesh,t,c,dt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 4-stage SSP RK3 numerical integration

    Parameters
    ----------
        mesh : Mesh dictionary
        t    : Current simulation time (currently unused)

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """

    ftn=np.array([mesh["Wj"],mesh["HUj"],mesh["HVj"]])                                             #0th step
    dt=forward_euler_weno(mesh,t,2,dt,0)[1]                                                        #1st step, forward euler of last step with 1/2 dt, saves dt
    forward_euler_weno(mesh,t,2,dt,cm)                                                             #2nd step, forward euler of last step with 1/2 dt
    mesh["Wj"],mesh["HUj"],mesh["HVj"]=2*ftn/3+(np.array(forward_euler_weno(mesh,t,2,dt,0)[2:]))/3 #3rd step, forces mesh to be 2/3 0th step + 1/3 forward euler of last step with 1/2 dt
    forward_euler_weno(mesh,t,2,dt,cm)                                                             #4th step, forward euler of last step with 1/2 dt

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def rk_instep_weno(mesh,t,rkc=1,rkdt=0,cm=0):

    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 1 step forward Euler numerical integration using quadratic reconstruction at each cell.

    Parameters
    ----------
        mesh : Mesh dictionary with the following fields:
          "Wj"  : Cell-centered water level, [1,nElem]
          "HUj" : Cell-centered Flux over X, [1,nElem]
          "HVj" : Cell-centered Flux over Y, [1,nElem]
          "ljk" : Side lengths for each triangle, [3,nElem]
          "rjk" : Side altitudes for each triangle, [3,nElem]
          "Tj"  : Area of the j-th Triangle, [1,nElem]
          "nx"  : Unit-Normal component over X, [3,nElem]
          "ny"  : Unit-Normal component over Y, [3,nElem]
          "g"   : Gravity constant, [1,1]
          "n"   : Manning's number, [1,nElem]
          "DRY" : Maximum water level to be consider as zero, [1,1]
        t    : Current simulation time (currently unused)
        rkc  : Runge-Kutta constant for use in RK integration
        rkdt : Timestep to be used. If not provided, function calculates it with CFL condition

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """
    #UNPACK MESH VARIABLES
    Bmj1 = mesh["Bmj1"]
    Bmj2 = mesh["Bmj2"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    ov=mesh["no_2nd_neigh"] | mesh["no_1st_neigh"]
    choice = mesh["cChoose1"]
    
    #PIECEWISE RECONSTRUCTION
    wrecon, wx, wy = PieceWiseReconstruction.weno2(mesh["Wj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])
    Wmj1, Wmj2  = PieceWiseReconstruction.set_weno_midpoint_values(mesh["Wj"],wrecon,xjk,yjk,ov)
    Wg1, Wg2, Wg3 = PieceWiseReconstruction.set_weno_in_values(wrecon, mesh["xg0"], mesh["xg1"], mesh["xg2"], mesh["yg0"], mesh["yg1"], mesh["yg2"],ov)
    wjk = PieceWiseReconstruction.set_weno_vertices_values(mesh["Wj"],wrecon,xjk,yjk,ov)
    recon = PieceWiseReconstruction.weno2(mesh["HUj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])[0]
    HUmj1, HUmj2  = PieceWiseReconstruction.set_weno_midpoint_values(mesh["HUj"],recon,xjk,yjk,ov)
    recon = PieceWiseReconstruction.weno2(mesh["HVj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])[0]
    HVmj1, HVmj2  = PieceWiseReconstruction.set_weno_midpoint_values(mesh["HVj"],recon,xjk,yjk,ov)

    #Well Balanced wet-dry Reconstruction
    Hmj1     = CentralUpwindMethod.set_water_depth(Wmj1, Bmj1, mesh["Constants"]["Dry"])
    Hmj2     = CentralUpwindMethod.set_water_depth(Wmj2, Bmj2, mesh["Constants"]["Dry"])
    #wjk,Wmj1,Wmj2,Hmj1,Hmj2,wxc,wyc,wreconc = PieceWiseReconstruction.set_linear_well_balanced_wet_dry2(wjk, mesh["Bjk"], Wmj1, Wmj2, Hmj1, Hmj2, Bmj1, Bmj2, Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy,wrecon)
    wxc=wx
    wyc=wy
    #Wg1, Wg2, Wg3 = PieceWiseReconstruction.set_weno_in_values(wreconc, mesh["xg0"], mesh["xg1"], mesh["xg2"], mesh["yg0"], mesh["yg1"], mesh["yg2"],ov)

    #This should update the inside gaussian points for W too!
    
    #Computation of the desingularizated velocities
    HUmj1, umj1  = CentralUpwindMethod.midvelocity(HUmj1, Hmj1, mesh["Constants"]["Dry"])
    HVmj1, vmj1  = CentralUpwindMethod.midvelocity(HVmj1, Hmj1, mesh["Constants"]["Dry"])

    HUmj2, umj2  = CentralUpwindMethod.midvelocity(HUmj2, Hmj2, mesh["Constants"]["Dry"])
    HVmj2, vmj2  = CentralUpwindMethod.midvelocity(HVmj2, Hmj2, mesh["Constants"]["Dry"])

    #Calculate the outer values for each variable
    Wmj1jk = Wmj1.take(mesh["jk"])
    Wmj2jk = Wmj2.take(mesh["jk"])
    Wmj1out = np.where(choice==0, Wmj1jk, Wmj2jk)
    Wmj2out = np.where(choice==1, Wmj1jk, Wmj2jk)

    Hmj1jk = Hmj1.take(mesh["jk"])
    Hmj2jk = Hmj2.take(mesh["jk"])
    Hmj1out = np.where(choice==0, Hmj1jk, Hmj2jk)
    Hmj2out = np.where(choice==1, Hmj1jk, Hmj2jk)

    umj1jk = umj1.take(mesh["jk"])
    umj2jk = umj2.take(mesh["jk"])
    umj1out = np.where(choice==0, umj1jk, umj2jk)
    umj2out = np.where(choice==1, umj1jk, umj2jk)

    vmj1jk = vmj1.take(mesh["jk"])
    vmj2jk = vmj2.take(mesh["jk"])
    vmj1out = np.where(choice==0, vmj1jk, vmj2jk)
    vmj2out = np.where(choice==1, vmj1jk, vmj2jk)

    HUmj1jk = HUmj1.take(mesh["jk"])
    HUmj2jk = HUmj2.take(mesh["jk"])
    HUmj1out = np.where(choice==0, HUmj1jk, HUmj2jk)
    HUmj2out = np.where(choice==1, HUmj1jk, HUmj2jk)

    HVmj1jk = HVmj1.take(mesh["jk"])
    HVmj2jk = HVmj2.take(mesh["jk"])
    HVmj1out = np.where(choice==0, HVmj1jk, HVmj2jk)
    HVmj2out = np.where(choice==1, HVmj1jk, HVmj2jk)

    #Computes the one-sided propagation velocity
    a_out, a_in = CentralUpwindMethod.one_sided_speed2(Hmj1, umj1, vmj1, Hmj2, umj2, vmj2, Hmj1out, umj1out, vmj1out, Hmj2out,umj2out,vmj2out, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    #Computes the maximum time step
    dt = max_time_step_gl(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"],mesh["GhostCells"]) if rkdt == 0 else rkdt

    #NUMERICAL INTEGRATION TO NEXT TIME STEP
    #Bottom Friction Values [ T  O    D  O !!!!!!!! ]
    Fr = CentralUpwindMethod.friction_term(0.5*(HUmj1+HUmj2), 0.5*(HVmj1+HVmj2), mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    #Source Term Values
    Sx, Sy = CentralUpwindMethod.source_term2(Hmj1, Hmj2, Wg1, Wg2, Wg3, mesh["Bg0"], mesh["Bg1"], mesh["Bg2"], wxc(mesh["xg0"],mesh["yg0"],ov), wxc(mesh["xg1"],mesh["yg1"],ov), wxc(mesh["xg2"],mesh["yg2"],ov), wyc(mesh["xg0"],mesh["yg0"],ov), wyc(mesh["xg1"],mesh["yg1"],ov), wyc(mesh["xg2"],mesh["yg2"],ov), mesh["ljk"], mesh["Tj"], mesh["nx"], mesh["ny"], mesh["Constants"]["Gravity"])
    fx, fy = CentralUpwindMethod.coriolis(mesh["HUj"],mesh["HVj"],mesh["Constants"]["coriolis"])

    #Water level surface (Forward Euler)
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.water_level_function2(Wmj1, HUmj1, HVmj1, Wmj1out, HUmj1out, HVmj1out, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.water_level_function2(Wmj2, HUmj2, HVmj2, Wmj2out, HUmj2out, HVmj2out, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    fwj = (dt*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"]) #Acá iría el RK3
    mesh["Wj"] = mesh["Wj"] + fwj/rkc

    #Flux along X-direction
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.flux_function_x2(Hmj1, HUmj1, HVmj1, Hmj1out, HUmj1out, HVmj1out, a_in, a_out, umj1, umj1out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.flux_function_x2(Hmj2, HUmj2, HVmj2, Hmj2out, HUmj2out, HVmj2out, a_in, a_out, umj2, umj2out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    fhu = (dt*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"] + dt*Sx + dt*fx) #acá igual
    mesh["HUj"] = mesh["HUj"] + fhu/rkc

    #Flux along Y-direction
    Uin1, Uout1, FUin1, FUout1, GUin1, GUout1 = CentralUpwindMethod.flux_function_y2(Hmj1, HUmj1, HVmj1, Hmj1out, HUmj1out, HVmj1out, a_in, a_out, vmj1, vmj1out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    Uin2, Uout2, FUin2, FUout2, GUin2, GUout2 = CentralUpwindMethod.flux_function_y2(Hmj2, HUmj2, HVmj2, Hmj2out, HUmj2out, HVmj2out, a_in, a_out, vmj2, vmj2out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    fhv =  (dt*( 0.5*((Uout1 - Uin1) - (FUin1 + FUout1) - (GUin1 + GUout1)) + 0.5*((Uout2 - Uin2) - (FUin2 + FUout2) - (GUin2 + GUout2)) )/mesh["Tj"] + dt*Sy + dt*fy) #acá igual
    mesh["HVj"] = mesh["HVj"] + fhv/rkc

    #Update value using the friction
    #mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    #mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)
    
    #Computed for ploting purposes
    #mesh["Hj"] = np.maximum(mesh["Wj"] - mesh["Bj"], mesh["Constants"]["Dry"])

    BoundaryConditions.impose(mesh)

    return wjk, dt, fwj, fhu, fhv

def rk_instep_minmod(mesh,t,rkc=1,rkdt=0,cm=0):
    """
    Computes a single step for the updated variables Wj, Wjk, HUj, HVj by means of 1 step forward Euler numerical integration

    Parameters
    ----------
        mesh : Mesh dictionary with the following fields:
          "Wj"  : Cell-centered water level, [1,nElem]
          "HUj" : Cell-centered Flux over X, [1,nElem]
          "HVj" : Cell-centered Flux over Y, [1,nElem]
          "ljk" : Side lengths for each triangle, [3,nElem]
          "rjk" : Side altitudes for each triangle, [3,nElem]
          "Tj"  : Area of the j-th Triangle, [1,nElem]
          "nx"  : Unit-Normal component over X, [3,nElem]
          "ny"  : Unit-Normal component over Y, [3,nElem]
          "g"   : Gravity constant, [1,1]
          "n"   : Manning's number, [1,nElem]
          "DRY" : Maximum water level to be consider as zero, [1,1]
        t    : Current simulation time (currently unused)
        rkc  : Runge-Kutta constant for use in RK integration
        rkdt : Timestep to be used. If not provided, function calculates it with CFL condition

    Returns
    -------
        Wjk : Updated vertices water level,      [3, nElem]
        dt  : Maximum allowed time step,         [1,     1] 
        Wj  : Updated Cell-centered water level, [1, nElem]
        HUj : Updated Cell-centered Flux over X, [1, nElem]
        HVj : Updated Cell-centered Flux over Y, [1, nElem]
    """
    #UNPACK MESH VARIABLES
    Bmj = mesh["Bmj"]
    Bj = mesh["Bj"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    #PIECEWISE RECONSTRUCTION
    wx, wy = PieceWiseReconstruction.minmod(mesh["Wj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    Wmj  = PieceWiseReconstruction.set_linear_midpoint_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    wjk = PieceWiseReconstruction.set_linear_vertices_values(mesh["Wj"], wx, wy, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HUj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HUmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HUj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)
    Lx, Ly = PieceWiseReconstruction.minmod(mesh["HVj"], mesh["xj"], mesh["yj"],xjk,yjk,mesh["Neighboors"],mesh["Constants"]["Tolerance"],mesh["Constants"]["theta"])
    HVmj = PieceWiseReconstruction.set_linear_midpoint_values(mesh["HVj"], Lx, Ly, mesh["xj"], mesh["yj"], xjk, yjk)

    #Well Balanced wet-dry Reconstruction
    Hmj     = CentralUpwindMethod.set_water_depth(Wmj, Bmj, mesh["Constants"]["Dry"])
    wjk,Wmj,Hmj,wx,wy = PieceWiseReconstruction.set_linear_well_balanced_wet_dry(wjk, mesh["Bjk"], Wmj, Hmj, mesh["Bmj"], Bj, mesh["Wj"], mesh["xj"], mesh["yj"], xjk, yjk, mesh["Constants"]["Tolerance"], mesh["Constants"]["Dry"], mesh["jk"], mesh["GhostCells"],wx,wy)
    
    
    #Computation of the desingularizated velocities
    HUmj, umj  = CentralUpwindMethod.midvelocity(HUmj, Hmj, mesh["Constants"]["Dry"])
    HVmj, vmj  = CentralUpwindMethod.midvelocity(HVmj, Hmj, mesh["Constants"]["Dry"])

    #Computes the one-sided propagation velocity
    a_out, a_in = CentralUpwindMethod.one_sided_speed(Hmj, umj, vmj, mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])

    #Computes the maximum time step
    dt = max_time_step(a_in, a_out, mesh["rjk"], mesh["Constants"]["CFL"]) if rkdt == 0 else rkdt

    #NUMERICAL INTEGRATION TO NEXT TIME STEP
    #Bottom Friction Values
    #Fr = CentralUpwindMethod.friction_term(HUmj, HVmj, mesh["Wj"], mesh["Bj"], mesh["Constants"]["Roughness"], mesh["Constants"]["Gravity"], mesh["Constants"]["Dry"])

    #Source Term Values
    Sx, Sy = CentralUpwindMethod.source_term(Hmj, wx, wy, mesh["ljk"], mesh["Tj"], mesh["nx"], mesh["ny"], mesh["Constants"]["Gravity"])

    #Water level surface (Forward Euler)
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.water_level_function(Wmj, HUmj, HVmj, a_in, a_out, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"])
    fwj = (dt*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]) #Acá iría el RK3
    mesh["Wj"] = mesh["Wj"]  + fwj/rkc

    #Flux along X-direction
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.flux_function_x(Hmj, HUmj, HVmj, a_in, a_out, umj, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    fhu = (dt*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]  + dt*Sx) #acá igual
    mesh["HUj"] = mesh["HUj"] + fhu/rkc

    #Flux along Y-direction
    Uin, Uout, FUin, FUout, GUin, GUout = CentralUpwindMethod.flux_function_y(Hmj, HUmj, HVmj, a_in, a_out, vmj, mesh["ljk"], mesh["nx"], mesh["ny"], mesh["jk"], mesh["Constants"]["Gravity"])
    fhv =  (dt*( (Uout - Uin) - (FUin + FUout) - (GUin + GUout) )/mesh["Tj"]  + dt*Sy) #acá igual
    mesh["HVj"] = mesh["HVj"] + fhv/rkc

    #Update value using the friction
    #mesh["HUj"] = mesh["HUj"]/(1.0 + mesh["dt"]*Fr)
    #mesh["HVj"] = mesh["HVj"]/(1.0 + mesh["dt"]*Fr)
    
    #Computed for ploting purposes
    #mesh["Hj"] = np.maximum(mesh["Wj"] - mesh["Bj"], mesh["Constants"]["Dry"])

    BoundaryConditions.impose(mesh)

    return wjk, dt, fwj, fhu, fhv

def runge_kutta4_weno(mesh,t,c,dt=0,cm=0):

    yn1 = mesh["Wj"]
    yn2 = mesh["HUj"]
    yn3 = mesh["HVj"]
    _          ,dt,k11,k12,k13=rk_instep_weno(mesh,t,2,dt,0)      #1st step, calculates fluxes k1 of state yn, advances mesh to state yn + dt k1/2
    _          ,dt,k21,k22,k23=rk_instep_weno(mesh,t,2,dt,0)      #2nd step, calculates fluxes k2 of state yn + dt k1/2, advances mesh to state yn + dt k1/2 + dt k2/2
    mesh["Wj"]-=k11/2                                               #-|
    mesh["HUj"]-=k12/2                                              # |-- Corrects mesh state to yn + dt k2/2
    mesh["HVj"]-=k13/2                                              #-|
    _          ,dt,k31,k32,k33=rk_instep_weno(mesh,t,1,dt,0)      #3rd step, calculates fluxes k3 of state yn + dt k2/2, advances mesh to state yn + dt k2/2 + dt k3
    mesh["Wj"]-=k21/2                                               #-|
    mesh["HUj"]-=k22/2                                              # |-- Corrects mesh state to yn + dt k3
    mesh["HVj"]-=k23/2                                              #-|
    mesh["Wjk"],dt,k41,k42,k43=rk_instep_weno(mesh,t,1,dt,0)      #4th step, calculates fluxes k4 of state yn + dt k3, advances mesh to state yn + dt k4 (to be corrected)

    mesh["Wj"]  = yn1 + (k11+2*k21+2*k31+k41)/6     # -|
    mesh["HUj"] = yn2 + (k12+2*k22+2*k32+k42)/6     #  |-- Corrects mesh with weighted RK4 average 
    mesh["HVj"] = yn3 + (k13+2*k23+2*k33+k43)/6     # -|

    mesh["dt"] = dt

    #Coordinates = mesh["Coordinates"]
    #Elements    = mesh["Elements"]
    #xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    #yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    #ov=mesh["no_2nd_neigh"] | mesh["no_1st_neigh"]
    #choice = mesh["cChoose1"]
    
    #WENO RECONSTRUCTION FOR PLOTTING
    #wrecon, wx, wy = PieceWiseReconstruction.weno2(mesh["Wj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])
    #mesh["Wjk"] =  PieceWiseReconstruction.set_weno_vertices_values(mesh["Wj"],wrecon,xjk,yjk,ov)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]

def runge_kutta4(mesh,t,c,dt=0,cm=0):

    yn1 = mesh["Wj"]
    yn2 = mesh["HUj"]
    yn3 = mesh["HVj"]
    _          ,dt,k11,k12,k13=rk_instep_minmod(mesh,t,2,dt,0)      #1st step, calculates fluxes k1 of state yn, advances mesh to state yn + dt k1/2
    _          ,dt,k21,k22,k23=rk_instep_minmod(mesh,t,2,dt,0)      #2nd step, calculates fluxes k2 of state yn + dt k1/2, advances mesh to state yn + dt k1/2 + dt k2/2
    mesh["Wj"]-=k11/2                                               #-|
    mesh["HUj"]-=k12/2                                              # |-- Corrects mesh state to yn + dt k2/2
    mesh["HVj"]-=k13/2                                              #-|
    _          ,dt,k31,k32,k33=rk_instep_minmod(mesh,t,1,dt,0)      #3rd step, calculates fluxes k3 of state yn + dt k2/2, advances mesh to state yn + dt k2/2 + dt k3
    mesh["Wj"]-=k21/2                                               #-|
    mesh["HUj"]-=k22/2                                              # |-- Corrects mesh state to yn + dt k3
    mesh["HVj"]-=k23/2                                              #-|
    mesh["Wjk"],dt,k41,k42,k43=rk_instep_minmod(mesh,t,1,dt,0)      #4th step, calculates fluxes k4 of state yn + dt k3, advances mesh to state yn + dt k3 + dt k4 (to be corrected)

    mesh["Wj"]  = yn1 + (k11+2*k21+2*k31+k41)/6     # -|
    mesh["HUj"] = yn2 + (k12+2*k22+2*k32+k42)/6     #  |-- Corrects mesh with weighted RK4 average 
    mesh["HVj"] = yn3 + (k13+2*k23+2*k33+k43)/6     # -|

    mesh["dt"] = dt
    #Coordinates = mesh["Coordinates"]
    #Elements    = mesh["Elements"]
    #xjk         = Coordinates[0,Elements]       # Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    #yjk         = Coordinates[1,Elements]       # Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    #ov=mesh["no_2nd_neigh"] | mesh["no_1st_neigh"]
    #choice = mesh["cChoose1"]
    
    #WENO RECONSTRUCTION FOR PLOTTING
    #wrecon, wx, wy = PieceWiseReconstruction.weno2(mesh["Wj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])
    #mesh["Wjk"] =  PieceWiseReconstruction.set_weno_vertices_values(mesh["Wj"],wrecon,xjk,yjk,ov)

    return mesh["Wjk"], mesh["dt"], mesh["Wj"], mesh["HUj"], mesh["HVj"]