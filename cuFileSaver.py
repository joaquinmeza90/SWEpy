import os
import cupy as cp
import numpy as np

def trunc(values, decs=0):
    return cp.trunc(values*10**decs)/(10**decs)

def save_bathymetry(mesh):
    """
    This function generates the bathymetry to be displayed in Paraview

    Parameters
    ----------
        mesh : Mesh dictionary with the simulation information

    Returns
    -------
        writes to disk the Bathymetry.vtu file
    """
    #Unpack Variables
    Nodes = cp.asnumpy(mesh["Coordinates"].T)
    Elems = cp.asnumpy(mesh["Elements"].T)
    Bsurf = cp.asnumpy(trunc(mesh["Bjk"].T,16))
    
    #Remove the ghost cells 
    nGhost = cp.asnumpy(mesh["GhostCells"].shape[1])
    Elems = Elems[:-nGhost,:]
    nElems = Elems.shape[0]

    #Paraview file name
    filename = "Paraview/Bathymetry.vtu"
    filename = os.path.join(mesh["FilePath"]["Directory"], filename)

    #Opens the ParaView File 
    Paraviewfile = open(filename, "w+")

    #Creates the VTU file for Paraview
    Paraviewfile.write("<?xml version=\"1.0\"?>\n")
    Paraviewfile.write("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"BigEndian\">\n")
    Paraviewfile.write("  <UnstructuredGrid>\n")
    Paraviewfile.write("    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n" % (3*nElems, nElems))

    #Writes Node coordinates
    Paraviewfile.write("      <Points>\n")
    Paraviewfile.write("        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
    for eTag in Elems:
        Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[0],0], Nodes[eTag[0],1], 0.0)) 
        Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[1],0], Nodes[eTag[1],1], 0.0)) 
        Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[2],0], Nodes[eTag[2],1], 0.0)) 
    Paraviewfile.write("        </DataArray>\n")
    Paraviewfile.write("      </Points>\n")

    #Writes Elements connectivity array
    Paraviewfile.write("      <Cells>\n")
    Paraviewfile.write("        <DataArray type=\"Int64\" Name=\"connectivity\" Format=\"ascii\">")
    for eTag in range(nElems):
        Paraviewfile.write("        %d %d %d\n" % (3*eTag, 3*eTag+1, 3*eTag+2))
    Paraviewfile.write("        </DataArray>\n")

    #Writes the Elements offset
    nOffsets = 0
    Paraviewfile.write("        <DataArray type=\"Int64\" Name=\"offsets\" Format=\"ascii\">\n       ")
    for eTag in range(nElems):
        nOffsets += 3
        Paraviewfile.write(" %d" % nOffsets)
    Paraviewfile.write("\n")
    Paraviewfile.write("        </DataArray>\n")

    #Writes the Elements type
    Paraviewfile.write("        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n       ")
    for eTag in range(nElems):
        Paraviewfile.write(" 5")
    Paraviewfile.write("\n")
    Paraviewfile.write("        </DataArray>\n")
    Paraviewfile.write("      </Cells>\n")
    
    #Writes the Bathymetry deformation
    Paraviewfile.write("      <PointData>\n")
    Paraviewfile.write("        <DataArray type=\"Float64\" Name=\"Bathymetry\" NumberOfComponents=\"3\" ComponentName0=\"Bx\" ComponentName1=\"By\" ComponentName2=\"Bz\" format=\"ascii\">\n")
    for bValues in Bsurf:
        Paraviewfile.write("         0.000 0.000 %E\n" % bValues[0])
        Paraviewfile.write("         0.000 0.000 %E\n" % bValues[1])
        Paraviewfile.write("         0.000 0.000 %E\n" % bValues[2])
    Paraviewfile.write("        </DataArray>\n")
    Paraviewfile.write("      </PointData>\n")

    #Close the ParaView VTU file
    Paraviewfile.write("    </Piece>\n")
    Paraviewfile.write("  </UnstructuredGrid>\n")
    Paraviewfile.write("</VTKFile>")
    return

def save_animation(mesh, t=0, force=False):
    """
    This function generates the water level and fluxes to be displayed in Paraview

    Parameters
    ----------
        mesh : Mesh dictionary with the simulation information

    Returns
    -------
        writes to disk the Solution.k.vtu file
    """
    #Paraview file name is written for sampling time
    if (mesh["s"]*mesh["Constants"]["dt_save"] <= t) or force:
        #Unpack Variables
        Nodes = mesh["Coordinates"].T
        Elems = mesh["Elements"].T
        Bsurf = mesh["Bjk"].T
        HUj = mesh["HUj"]
        HVj = mesh["HVj"]
        
        #Remove the ghost cells 
        nGhost = mesh["GhostCells"].shape[1]
        Elems = Elems[:-nGhost,:]
        nElems = Elems.shape[0]
        
        TOL = 10*mesh["Constants"]["Dry"]
        Wjk = mesh["Wjk"]
        Wj = mesh["Wj"]
        Hj  = cp.maximum(mesh["Wj"] - mesh["Bj"], 0.01*TOL)
        nElems = Elems.shape[0]

        Hj = cp.asnumpy(Hj)

        #Elements that are not dry
        rednum = 0
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                rednum += 1

        #Turn into numpy array to avoid synch problems
        Nodes = cp.asnumpy(Nodes)
        Elems = cp.asnumpy(Elems)
        Bsurf = cp.asnumpy(Bsurf)
        HUj = cp.asnumpy(HUj)
        HVj = cp.asnumpy(HVj)
        Wjk = cp.asnumpy(trunc(Wjk,10))
        Wj = cp.asnumpy(trunc(Wj,10))

        filename = "Paraview/Simulation_" + str(mesh["s"]) + ".vtu" if not force else "Paraview/Simulation_" + str(mesh["s"]) + "forced.vtu"
        filename = os.path.join(mesh["FilePath"]["Directory"], filename)

        #Opens the ParaView File 
        Paraviewfile = open(filename, "w+")

        #Creates the VTU file for Paraview
        Paraviewfile.write("<?xml version=\"1.0\"?>\n")
        Paraviewfile.write("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"BigEndian\">\n")
        Paraviewfile.write("  <UnstructuredGrid>\n")
        Paraviewfile.write("    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n" % (3*nElems, rednum))

        #Writes Node coordinates
        Paraviewfile.write("      <Points>\n")
        Paraviewfile.write("        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
        for eTag in Elems:
            Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[0],0], Nodes[eTag[0],1], 0.0)) 
            Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[1],0], Nodes[eTag[1],1], 0.0)) 
            Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[2],0], Nodes[eTag[2],1], 0.0)) 
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </Points>\n")

        #Writes Elements connectivity array
        Paraviewfile.write("      <Cells>\n")
        Paraviewfile.write("        <DataArray type=\"Int64\" Name=\"connectivity\" Format=\"ascii\">\n")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                Paraviewfile.write("        %d %d %d\n" % (3*eTag, 3*eTag+1, 3*eTag+2))
        Paraviewfile.write("        </DataArray>\n")

        #Writes the Elements offset
        nOffsets = 0
        Paraviewfile.write("        <DataArray type=\"Int64\" Name=\"offsets\" Format=\"ascii\">\n       ")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                nOffsets += 3
                Paraviewfile.write(" %d" % nOffsets)
        Paraviewfile.write("\n")
        Paraviewfile.write("        </DataArray>\n")

        #Writes the Elements type
        Paraviewfile.write("        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n       ")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                Paraviewfile.write(" 5")
        Paraviewfile.write("\n")
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </Cells>\n")

        #Writes the Water level deformation
        Paraviewfile.write("      <PointData>\n")
        Paraviewfile.write("        <DataArray type=\"Float32\" Name=\"WaterSurface\" NumberOfComponents=\"3\" ComponentName0=\"Ujk\" ComponentName1=\"Vjk\" ComponentName2=\"Wjk\" format=\"ascii\">\n")
        for eTag in range(nElems):
            Paraviewfile.write("          0.000 0.000 %E\n" % Wjk[0,eTag]) #Wj[mm])
            Paraviewfile.write("          0.000 0.000 %E\n" % Wjk[1,eTag]) #Wj[mm])
            Paraviewfile.write("          0.000 0.000 %E\n" % Wjk[2,eTag]) #Wj[mm])
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </PointData>\n")

        #Writes the Flux along X/Y directions and water level
        Paraviewfile.write("      <CellData>\n")
        Paraviewfile.write("        <DataArray type=\"Float32\" Name=\"SolvedVariables\" NumberOfComponents=\"3\" ComponentName0=\"HUj\" ComponentName1=\"HVj\" ComponentName2=\"Wj\" format=\"ascii\">\n")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                Paraviewfile.write("          %E %E %E\n" % (HUj[eTag], HVj[eTag], Wj[eTag]))
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </CellData>\n")

        #Close the ParaView VTU file
        Paraviewfile.write("    </Piece>\n")
        Paraviewfile.write("  </UnstructuredGrid>\n")
        Paraviewfile.write("</VTKFile>")

        #Update counter
        mesh["s"] += 1 if not force else 0

    return

def save_animation_analytic(mesh, t=0):
    """
    This function generates the water level and fluxes to be displayed in Paraview

    Parameters
    ----------
        mesh : Mesh dictionary with the simulation information

    Returns
    -------
        writes to disk the Solution.k.vtu file
    """
    #Paraview file name is written for sampling time
    if (mesh["s"]*mesh["Constants"]["dt_save"] <= t):
        #Unpack Variables
        Nodes = mesh["Coordinates"].T
        Elems = mesh["Elements"].T
        Bsurf = mesh["Bjk"].T
        HUj = mesh["HUj"]
        HVj = mesh["HVj"]
        
        #Remove the ghost cells 
        nGhost = int(mesh["GhostCells"].shape[1])
        Elems = Elems[:-nGhost,:]
        nElems = int(Elems.shape[0])
        
        TOL = 10*mesh["Constants"]["Dry"]
        Wjk = mesh["Wjk"]
        Wj = mesh["Wj"]
        Hj  = cp.maximum(mesh["Wj"] - mesh["Bj"], TOL)

        Hj = cp.asnumpy(Hj)

        #Elements that are not dry
        rednum = 0
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                rednum += 1

        #Turn into numpy array to avoid synch problems
        Nodes = cp.asnumpy(Nodes)
        Elems = cp.asnumpy(Elems)
        Bsurf = cp.asnumpy(Bsurf)
        HUj = cp.asnumpy(HUj)
        HVj = cp.asnumpy(HVj)
        Wjk = cp.asnumpy(Wjk)
        Wj = cp.asnumpy(Wj)
                
        filename = "Paraview/Analytic_" + str(mesh["s"]) + ".vtu"
        filename = os.path.join(mesh["FilePath"]["Directory"], filename)

        #Opens the ParaView File 
        Paraviewfile = open(filename, "w+")

        #Creates the VTU file for Paraview
        Paraviewfile.write("<?xml version=\"1.0\"?>\n")
        Paraviewfile.write("<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"BigEndian\">\n")
        Paraviewfile.write("  <UnstructuredGrid>\n")
        Paraviewfile.write("    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n" % (3*nElems, rednum))

        #Writes Node coordinates
        Paraviewfile.write("      <Points>\n")
        Paraviewfile.write("        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
        for eTag in Elems:
            Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[0],0], Nodes[eTag[0],1], 0.0)) 
            Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[1],0], Nodes[eTag[1],1], 0.0)) 
            Paraviewfile.write("        %E %E %E\n" % (Nodes[eTag[2],0], Nodes[eTag[2],1], 0.0)) 
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </Points>\n")

        #Writes Elements connectivity array
        Paraviewfile.write("      <Cells>\n")
        Paraviewfile.write("        <DataArray type=\"Int64\" Name=\"connectivity\" Format=\"ascii\">\n")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                Paraviewfile.write("        %d %d %d\n" % (3*eTag, 3*eTag+1, 3*eTag+2))
        Paraviewfile.write("        </DataArray>\n")

        #Writes the Elements offset
        nOffsets = 0
        Paraviewfile.write("        <DataArray type=\"Int64\" Name=\"offsets\" Format=\"ascii\">\n       ")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                nOffsets += 3
                Paraviewfile.write(" %d" % nOffsets)
        Paraviewfile.write("\n")
        Paraviewfile.write("        </DataArray>\n")

        #Writes the Elements type
        Paraviewfile.write("        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n       ")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                Paraviewfile.write(" 5")
        Paraviewfile.write("\n")
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </Cells>\n")

        #Writes the Water level deformation
        Paraviewfile.write("      <PointData>\n")
        Paraviewfile.write("        <DataArray type=\"Float32\" Name=\"WaterSurface\" NumberOfComponents=\"3\" ComponentName0=\"Ujk\" ComponentName1=\"Vjk\" ComponentName2=\"Wjk\" format=\"ascii\">\n")
        for (mm,wValue) in enumerate(Hj):
            Paraviewfile.write("        0.000 0.000 %E\n" % Wjk[0,mm]) #Wj[mm])
            Paraviewfile.write("        0.000 0.000 %E\n" % Wjk[1,mm]) #Wj[mm])
            Paraviewfile.write("        0.000 0.000 %E\n" % Wjk[2,mm]) #Wj[mm])
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </PointData>\n")

        #Writes the Flux along X/Y directions and water level
        Paraviewfile.write("      <CellData>\n")
        Paraviewfile.write("        <DataArray type=\"Float32\" Name=\"SolvedVariables\" NumberOfComponents=\"3\" ComponentName0=\"HUj\" ComponentName1=\"HVj\" ComponentName2=\"Wj\" format=\"ascii\">\n")
        for eTag in range(nElems):
            if Hj[eTag] > TOL:
                Paraviewfile.write("          %E %E %E\n" % (HUj[eTag], HVj[eTag], Wj[eTag]))
        Paraviewfile.write("        </DataArray>\n")
        Paraviewfile.write("      </CellData>\n")

        #Close the ParaView VTU file
        Paraviewfile.write("    </Piece>\n")
        Paraviewfile.write("  </UnstructuredGrid>\n")
        Paraviewfile.write("</VTKFile>")

        #Update counter
        mesh["s"] += 1

    return

def save_error_calc(mesh):

    filename = "L2Error.txt"
    filename = os.path.join(mesh["FilePath"]["Directory"], filename)
     
    Textfile = open(filename, "w+")

    Textfile.write("#T  L2Error\n")

    for i in range(mesh["s"]):
        Textfile.write(str(i*mesh["Constants"]["dt_save"])+"    "+str(mesh["L2Error"][i])+"\n")
        print("t = "+str(i*mesh["Constants"]["dt_save"])+", L2 Error = "+str(mesh["L2Error"][i]))

    Textfile.close()

    return

def save_last_Wj(mesh):
    filename = "Wj_xj_yj_at_Tmax.txt"
    filename = os.path.join(mesh["FilePath"]["Directory"], filename)

    Elems = cp.asnumpy(mesh["Elements"].T)

    #Remove the ghost cells 
    nGhost = int(mesh["GhostCells"].shape[1])
    Elems = Elems[:-nGhost,:]
    nElems = int(Elems.shape[0])

    Wj=cp.asnumpy(mesh["Wj"])
    xj=cp.asnumpy(mesh["xj"])
    yj=cp.asnumpy(mesh["yj"])
     
    Textfile = open(filename, "w+")

    Textfile.write("#Wj xj yj\n")

    for i in range(nElems):
        Textfile.write(str(Wj[i])+" "+str(xj[i])+" "+str(yj[i])+"\n")

    Textfile.close()

    return

def save_TS(mesh, t, gauges):
    #Saves water level and water depth at each gauge in [gauges] at each dt_save
    if (mesh["s"]*mesh["Constants"]["dt_save"] <= t):

        for gauge in gauges:
            filename="gauge_"+str(gauge)+".txt"
            filename = os.path.join(mesh["FilePath"]["Directory"], filename)
            txt = open(filename,"a")
            if type(gauge) is int:
                txt.write(str(trunc(t,10))+" "+str(mesh["Wj"][gauge])+" "+str(mesh["Wj"][gauge]-mesh["Bj"][gauge])+"\n")
            else:
                txt.write(str(trunc(t,10))+" "+str(mesh["Wj"][mesh["Wj"]>mesh["Bj"]+mesh["Constants"]["Dry"]].max())+" "+str(mesh["Wj"][mesh["Wj"]>mesh["Bj"]+mesh["Constants"]["Dry"]].max()-mesh["Bj"][mesh["Wj"][mesh["Wj"]>mesh["Bj"]+mesh["Constants"]["Dry"]].argmax()])+"\n")
            txt.close()


        #Update counter
        mesh["s"] += 1

    return