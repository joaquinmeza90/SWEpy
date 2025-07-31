import os
import inspect
import cupy as np
from cupyx.scipy.interpolate import CloughTocher2DInterpolator

import cuFileSaver as FileSaver
import cuPieceWiseReconstruction as PieceWiseReconstruction

def create_stencils():
    stencils1=[]
    stencils2=[]
    idx=(0,1,2)
    for n in idx:
        ln=[]
        ln.append((n,0))
        ln.append((n,1))
        ln.append((n,-1))
        r = set(idx)-{n}
        for i in r:
            li=ln.copy()
            li.append((i,-1))
            for j in [0,1]:
                lj=li.copy()
                lj.append((i,j))
                stencils1.append(tuple(lj))
    for n in idx:
        ln=[]
        ln.append((n,-1))
        r=list(set(idx)-{n})
        ln.append((r[0],-1))
        ln.append((r[1],-1))
        for i in [0,1]:
            for j in [0,1]:
                lp=ln.copy()
                lp.append((r[0],i))
                lp.append((r[1],j))
                stencils2.append(tuple(lp))
    return stencils1,stencils2

def create_empty_mesh():
    """
    Creates an empty mesh to store simulation results

    Parameters
    ----------
        None

    Returns
    -------
        mesh : Mesh dictionary with specified fields 
    """
    #Defines an empty dictionary with required fields
    mesh = {
        "xj"    : [],
        "yj"    : [],
        "nx"    : [],
        "ny"    : [],
        "ljk"   : [],
        "Tj"    : [],
        "rjk"   : [],
        "Wj"    : [],
        "HUj"   : [],
        "HVj"   : [],
        "Bj"    : [],
        "Hj"    : [],
        "Bjk"   : [],
        "Wjk"   : [],
        "Bmj"   : [],
        "jk"    : [],
        "s"     : 0,
        "Coordinates" : [],
        "Elements"    : [],
        "Neighboors"  : [],
        "GhostCells"  : [],
        "L2Error"     : [],
        "Constants" : {
            "Dry"      : 1.0E-6,
            "CFL"      : 0.25,
            "Tmax"     : 10.0,
            "dt_save"  : 1.00,
            "Gravity"  : 9.81,
            "Tolerance": 1.0E-6,
            "Roughness": 0.00,
            "dL"       : 1,
            "theta"    : 1,
            "coriolis" : 0
        },
        "InitialConditions" : {
            "WaterLevel" : "",
            "Discharge"  : ""
        },
        "FilePath" : {
            "Directory"    : os.getcwd(),
            "Bathymetry"   : "",
            "Coordinates"  : "",
            "Sides"        : "",
            "Elements"     : "",
            "Neighboors"   : "",
            "GhostCells"   : "",
            "Configuration": ""
        }
    }

    return mesh

def global_index_selection(Neighboors, Sides):
    nElems = Neighboors.shape[1]

    col = Neighboors.copy()
    ind = col<0
    aux = np.array([range(nElems), range(nElems), range(nElems)])
    col[ind] = aux[ind]

    row = Sides.copy()
    ind = row<0
    one = np.ones(nElems, dtype=int)
    aux = np.array([0*one,one,2*one])
    row[ind] = aux[ind]

    #Indeces
    position = row*nElems + col
    
    return position

def neighbor2(mesh):

    nElems=len(mesh["Elements"].T) #Calculate number of elements

    aux=np.array([[True,True,True],[True,True,True],[True,True,True]]) #initialize cell mask
    auxstack=np.tile(aux,(nElems,1,1)) #initialize cell mask for each cell

    enum=np.tile(np.array([0,1,2]),(nElems,1)) #initialize side enumeration for each cell

    ns=mesh["NeighSides"] #unpack neighbor sides

    auxstack[np.array(range(nElems))[:,None],enum,ns.T]=False #for each cell's mask, assign false to the position of itself on its neighbors' neighbors

    faux_neighs=np.copy(mesh["Neighboors"]) #copy the neighbors array

    #faux_neighs[faux_neighs<0]=np.where(faux_neighs<0)[1] #Make non-mesh neighbors of ghost cells equal to themselves, so second neighbors of border cells are equal to its corresponding ghost cell

    mesh["Neighs2"]=faux_neighs.T[faux_neighs.T][auxstack].reshape((nElems,3,2)) #for each cell, we get a (3,2)-array with a row for each of its neighbors containing the neighbors of that neighbor that are not the cell itself

    mesh["no_2nd_neigh"]=(np.any(np.isin(mesh["Neighboors"],np.hstack((mesh["GhostCells"][0,:],-1))),axis=0))

    mesh["no_1st_neigh"]=(np.any(np.isin(mesh["Neighboors"],np.hstack((-1,-1))),axis=0))

    return

def second_moments(mesh):
    x0 = mesh["Coordinates"][0,:][mesh["Elements"][0,:]]
    x1 = mesh["Coordinates"][0,:][mesh["Elements"][1,:]]
    x2 = mesh["Coordinates"][0,:][mesh["Elements"][2,:]]
    y0 = mesh["Coordinates"][1,:][mesh["Elements"][0,:]]
    y1 = mesh["Coordinates"][1,:][mesh["Elements"][1,:]]
    y2 = mesh["Coordinates"][1,:][mesh["Elements"][2,:]]

    mesh["Iy"]  = np.abs((1/12)*((x0*y1-x1*y0)*(x0**2+x0*x1+x1**2)+(x1*y2-x2*y1)*(x1**2+x1*x2+x2**2)+(x2*y0-x0*y2)*(x2**2+x2*x0+x0**2)))
    mesh["Ix"]  = np.abs((1/12)*((x0*y1-x1*y0)*(y0**2+y0*y1+y1**2)+(x1*y2-x2*y1)*(y1**2+y1*y2+y2**2)+(x2*y0-x0*y2)*(y2**2+y2*y0+y0**2)))
    mesh["Ixy"] = np.abs((1/24)*((x0*y1-x1*y0)*(x0*y1+2*x0*y0+2*x1*y1+x1*y0)+(x1*y2-x2*y1)*(x1*y2+2*x1*y1+2*x2*y2+x2*y1)+(x2*y0-x0*y2)*(x2*y0+2*x2*y2+2*x0*y0+x0*y2)))

    return

def second_moments2(mesh):
    mesh["Iy"] = (mesh["xg0"]**2+mesh["xg1"]**2+mesh["xg2"]**2)*mesh["Tj"]/3
    mesh["Ix"] = (mesh["yg0"]**2+mesh["yg1"]**2+mesh["yg2"]**2)*mesh["Tj"]/3
    mesh["Ixy"]= (mesh["xg0"]*mesh["yg0"]+mesh["xg1"]*mesh["yg1"]+mesh["xg2"]*mesh["yg2"])*mesh["Tj"]/3
    return

def quad_coefs(mesh,stencil):
    """
    Returns the quadratic reconstruction matrix coefficients given a stencil.

               /\ 
              /  \ 
             /  1 \ 
     X------X------X 
      \  2 / \  1 / \ 
       \  / c \  / 0 \ 
        \/_____\/_____\ 
        /\     /\       
       /  \ 0 /  \ 
      / 1  \ /  0 \ 
     V------V------\ 
     
    Input
    -----
        mesh    : Mesh dictionary
        stencil : (5,2)-Tuple in the form ((n1,m1),(n2,m2),(n3,m3),(n4,m4),(n5,m5)) to take the mi-th neighbor of the ni-th neighbor of each cell (excluding the cell itself)

    Returns
    -------
        coefij : (i,j)-th coefficient of the matrix A associated to the quadratic reconstruction problem Ax=B, where x=(Ux,Uy,Uxx,Uyy,Uxy), Bi = (U(xi)-U(x0))
    """

    l=mesh["NChoose"]

    x0 = mesh["xj"]
    x1 = mesh["xj"][l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    x2 = mesh["xj"][l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    x3 = mesh["xj"][l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    x4 = mesh["xj"][l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    x5 = mesh["xj"][l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    y0 = mesh["yj"]
    y1 = mesh["yj"][l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    y2 = mesh["yj"][l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    y3 = mesh["yj"][l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    y4 = mesh["yj"][l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    y5 = mesh["yj"][l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    T0 = mesh["Tj"]
    T1 = mesh["Tj"][l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    T2 = mesh["Tj"][l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    T3 = mesh["Tj"][l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    T4 = mesh["Tj"][l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    T5 = mesh["Tj"][l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    Iy0 = mesh["Iy"]
    Iy1 = mesh["Iy"][l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    Iy2 = mesh["Iy"][l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    Iy3 = mesh["Iy"][l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    Iy4 = mesh["Iy"][l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    Iy5 = mesh["Iy"][l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    Ix0 = mesh["Ix"]
    Ix1 = mesh["Ix"][l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    Ix2 = mesh["Ix"][l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    Ix3 = mesh["Ix"][l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    Ix4 = mesh["Ix"][l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    Ix5 = mesh["Ix"][l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    Ixy0 = mesh["Ixy"]
    Ixy1 = mesh["Ixy"][l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    Ixy2 = mesh["Ixy"][l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    Ixy3 = mesh["Ixy"][l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    Ixy4 = mesh["Ixy"][l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    Ixy5 = mesh["Ixy"][l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    coef11 = x1-x0
    coef12 = y1-y0
    coef13 = (Iy1/T1-Iy0/T0-2*x1*x0+2*x0**2)/2
    coef14 = (Ix1/T1-Ix0/T0-2*y1*y0+2*y0**2)/2
    coef15 = Ixy1/T1-Ixy0/T0-x0*y1-x1*y0+2*x0*y0
    coef21 = x2-x0
    coef22 = y2-y0
    coef23 = (Iy2/T2-Iy0/T0-2*x2*x0+2*x0**2)/2
    coef24 = (Ix2/T2-Ix0/T0-2*y2*y0+2*y0**2)/2
    coef25 = Ixy2/T2-Ixy0/T0-x0*y2-x2*y0+2*x0*y0
    coef31 = x3-x0
    coef32 = y3-y0
    coef33 = (Iy3/T3-Iy0/T0-2*x3*x0+2*x0**2)/2
    coef34 = (Ix3/T3-Ix0/T0-2*y3*y0+2*y0**2)/2
    coef35 = Ixy3/T3-Ixy0/T0-x0*y3-x3*y0+2*x0*y0
    coef41 = x4-x0
    coef42 = y4-y0
    coef43 = (Iy4/T4-Iy0/T0-2*x4*x0+2*x0**2)/2
    coef44 = (Ix4/T4-Ix0/T0-2*y4*y0+2*y0**2)/2
    coef45 = Ixy4/T4-Ixy0/T0-x0*y4-x4*y0+2*x0*y0
    coef51 = x5-x0
    coef52 = y5-y0
    coef53 = (Iy5/T5-Iy0/T0-2*x5*x0+2*x0**2)/2
    coef54 = (Ix5/T5-Ix0/T0-2*y5*y0+2*y0**2)/2
    coef55 = Ixy5/T5-Ixy0/T0-x0*y5-x5*y0+2*x0*y0

    return coef11,coef12,coef13,coef14,coef15,coef21,coef22,coef23,coef24,coef25,coef31,coef32,coef33,coef34,coef35,coef41,coef42,coef43,coef44,coef45,coef51,coef52,coef53,coef54,coef55

def inverse_coefs(coef11,coef12,coef13,coef14,coef15,coef21,coef22,coef23,coef24,coef25,coef31,coef32,coef33,coef34,coef35,coef41,coef42,coef43,coef44,coef45,coef51,coef52,coef53,coef54,coef55):

    D = coef15*coef24*coef33*coef42*coef51 - coef14*coef25*coef33*coef42*coef51 - coef15*coef23*coef34*coef42*coef51 +  coef13*coef25*coef34*coef42*coef51 +  coef14*coef23*coef35*coef42*coef51 - coef13*coef24*coef35*coef42*coef51 - coef15*coef24*coef32*coef43*coef51 +  coef14*coef25*coef32*coef43*coef51 +  coef15*coef22*coef34*coef43*coef51 - coef12*coef25*coef34*coef43*coef51 - coef14*coef22*coef35*coef43*coef51 +  coef12*coef24*coef35*coef43*coef51 +  coef15*coef23*coef32*coef44*coef51 - coef13*coef25*coef32*coef44*coef51 - coef15*coef22*coef33*coef44*coef51 +  coef12*coef25*coef33*coef44*coef51 +  coef13*coef22*coef35*coef44*coef51 - coef12*coef23*coef35*coef44*coef51 - coef14*coef23*coef32*coef45*coef51 +  coef13*coef24*coef32*coef45*coef51 +  coef14*coef22*coef33*coef45*coef51 - coef12*coef24*coef33*coef45*coef51 - coef13*coef22*coef34*coef45*coef51 +  coef12*coef23*coef34*coef45*coef51 - coef15*coef24*coef33*coef41*coef52 +  coef14*coef25*coef33*coef41*coef52 +  coef15*coef23*coef34*coef41*coef52 - coef13*coef25*coef34*coef41*coef52 - coef14*coef23*coef35*coef41*coef52 +  coef13*coef24*coef35*coef41*coef52 +  coef15*coef24*coef31*coef43*coef52 - coef14*coef25*coef31*coef43*coef52 - coef15*coef21*coef34*coef43*coef52 +  coef11*coef25*coef34*coef43*coef52 +  coef14*coef21*coef35*coef43*coef52 - coef11*coef24*coef35*coef43*coef52 - coef15*coef23*coef31*coef44*coef52 +  coef13*coef25*coef31*coef44*coef52 +  coef15*coef21*coef33*coef44*coef52 - coef11*coef25*coef33*coef44*coef52 - coef13*coef21*coef35*coef44*coef52 +  coef11*coef23*coef35*coef44*coef52 +  coef14*coef23*coef31*coef45*coef52 - coef13*coef24*coef31*coef45*coef52 - coef14*coef21*coef33*coef45*coef52 +  coef11*coef24*coef33*coef45*coef52 +  coef13*coef21*coef34*coef45*coef52 - coef11*coef23*coef34*coef45*coef52 +  coef15*coef24*coef32*coef41*coef53 - coef14*coef25*coef32*coef41*coef53 - coef15*coef22*coef34*coef41*coef53 +  coef12*coef25*coef34*coef41*coef53 +  coef14*coef22*coef35*coef41*coef53 - coef12*coef24*coef35*coef41*coef53 - coef15*coef24*coef31*coef42*coef53 +  coef14*coef25*coef31*coef42*coef53 +  coef15*coef21*coef34*coef42*coef53 - coef11*coef25*coef34*coef42*coef53 - coef14*coef21*coef35*coef42*coef53 +  coef11*coef24*coef35*coef42*coef53 +  coef15*coef22*coef31*coef44*coef53 - coef12*coef25*coef31*coef44*coef53 - coef15*coef21*coef32*coef44*coef53 +  coef11*coef25*coef32*coef44*coef53 +  coef12*coef21*coef35*coef44*coef53 - coef11*coef22*coef35*coef44*coef53 - coef14*coef22*coef31*coef45*coef53 +  coef12*coef24*coef31*coef45*coef53 +  coef14*coef21*coef32*coef45*coef53 - coef11*coef24*coef32*coef45*coef53 - coef12*coef21*coef34*coef45*coef53 +  coef11*coef22*coef34*coef45*coef53 - coef15*coef23*coef32*coef41*coef54 +  coef13*coef25*coef32*coef41*coef54 +  coef15*coef22*coef33*coef41*coef54 - coef12*coef25*coef33*coef41*coef54 - coef13*coef22*coef35*coef41*coef54 +  coef12*coef23*coef35*coef41*coef54 +  coef15*coef23*coef31*coef42*coef54 - coef13*coef25*coef31*coef42*coef54 - coef15*coef21*coef33*coef42*coef54 +  coef11*coef25*coef33*coef42*coef54 +  coef13*coef21*coef35*coef42*coef54 - coef11*coef23*coef35*coef42*coef54 - coef15*coef22*coef31*coef43*coef54 +  coef12*coef25*coef31*coef43*coef54 +  coef15*coef21*coef32*coef43*coef54 - coef11*coef25*coef32*coef43*coef54 - coef12*coef21*coef35*coef43*coef54 +  coef11*coef22*coef35*coef43*coef54 +  coef13*coef22*coef31*coef45*coef54 - coef12*coef23*coef31*coef45*coef54 - coef13*coef21*coef32*coef45*coef54 +  coef11*coef23*coef32*coef45*coef54 +  coef12*coef21*coef33*coef45*coef54 - coef11*coef22*coef33*coef45*coef54 +  coef14*coef23*coef32*coef41*coef55 - coef13*coef24*coef32*coef41*coef55 - coef14*coef22*coef33*coef41*coef55 +  coef12*coef24*coef33*coef41*coef55 +  coef13*coef22*coef34*coef41*coef55 - coef12*coef23*coef34*coef41*coef55 - coef14*coef23*coef31*coef42*coef55 +  coef13*coef24*coef31*coef42*coef55 +  coef14*coef21*coef33*coef42*coef55 - coef11*coef24*coef33*coef42*coef55 - coef13*coef21*coef34*coef42*coef55 +  coef11*coef23*coef34*coef42*coef55 +  coef14*coef22*coef31*coef43*coef55 - coef12*coef24*coef31*coef43*coef55 - coef14*coef21*coef32*coef43*coef55 +  coef11*coef24*coef32*coef43*coef55 +  coef12*coef21*coef34*coef43*coef55 - coef11*coef22*coef34*coef43*coef55 - coef13*coef22*coef31*coef44*coef55 +  coef12*coef23*coef31*coef44*coef55 +  coef13*coef21*coef32*coef44*coef55 - coef11*coef23*coef32*coef44*coef55 - coef12*coef21*coef33*coef44*coef55 +  coef11*coef22*coef33*coef44*coef55
    bool_aux = ( np.abs(D) < 1e-12 )
    D[bool_aux] = 1.0

    a11 = ( coef25*coef34*coef43*coef52 - coef24*coef35*coef43*coef52 - coef25*coef33*coef44*coef52 + coef23*coef35*coef44*coef52 + coef24*coef33*coef45*coef52 - coef23*coef34*coef45*coef52 - coef25*coef34*coef42*coef53 + coef24*coef35*coef42*coef53 + coef25*coef32*coef44*coef53 - coef22*coef35*coef44*coef53 - coef24*coef32*coef45*coef53 + coef22*coef34*coef45*coef53 + coef25*coef33*coef42*coef54 - coef23*coef35*coef42*coef54 - coef25*coef32*coef43*coef54 + coef22*coef35*coef43*coef54 + coef23*coef32*coef45*coef54 - coef22*coef33*coef45*coef54 - coef24*coef33*coef42*coef55 + coef23*coef34*coef42*coef55 + coef24*coef32*coef43*coef55 - coef22*coef34*coef43*coef55 - coef23*coef32*coef44*coef55 + coef22*coef33*coef44*coef55)/D
    a12 = (-coef15*coef34*coef43*coef52 + coef14*coef35*coef43*coef52 + coef15*coef33*coef44*coef52 - coef13*coef35*coef44*coef52 - coef14*coef33*coef45*coef52 + coef13*coef34*coef45*coef52 + coef15*coef34*coef42*coef53 - coef14*coef35*coef42*coef53 - coef15*coef32*coef44*coef53 + coef12*coef35*coef44*coef53 + coef14*coef32*coef45*coef53 - coef12*coef34*coef45*coef53 - coef15*coef33*coef42*coef54 + coef13*coef35*coef42*coef54 + coef15*coef32*coef43*coef54 - coef12*coef35*coef43*coef54 - coef13*coef32*coef45*coef54 + coef12*coef33*coef45*coef54 + coef14*coef33*coef42*coef55 - coef13*coef34*coef42*coef55 - coef14*coef32*coef43*coef55 + coef12*coef34*coef43*coef55 + coef13*coef32*coef44*coef55 - coef12*coef33*coef44*coef55)/D
    a13 = ( coef15*coef24*coef43*coef52 - coef14*coef25*coef43*coef52 - coef15*coef23*coef44*coef52 + coef13*coef25*coef44*coef52 + coef14*coef23*coef45*coef52 - coef13*coef24*coef45*coef52 - coef15*coef24*coef42*coef53 + coef14*coef25*coef42*coef53 + coef15*coef22*coef44*coef53 - coef12*coef25*coef44*coef53 - coef14*coef22*coef45*coef53 + coef12*coef24*coef45*coef53 + coef15*coef23*coef42*coef54 - coef13*coef25*coef42*coef54 - coef15*coef22*coef43*coef54 + coef12*coef25*coef43*coef54 + coef13*coef22*coef45*coef54 - coef12*coef23*coef45*coef54 - coef14*coef23*coef42*coef55 + coef13*coef24*coef42*coef55 + coef14*coef22*coef43*coef55 - coef12*coef24*coef43*coef55 - coef13*coef22*coef44*coef55 + coef12*coef23*coef44*coef55)/D
    a14 = (-coef15*coef24*coef33*coef52 + coef14*coef25*coef33*coef52 + coef15*coef23*coef34*coef52 - coef13*coef25*coef34*coef52 - coef14*coef23*coef35*coef52 + coef13*coef24*coef35*coef52 + coef15*coef24*coef32*coef53 - coef14*coef25*coef32*coef53 - coef15*coef22*coef34*coef53 + coef12*coef25*coef34*coef53 + coef14*coef22*coef35*coef53 - coef12*coef24*coef35*coef53 - coef15*coef23*coef32*coef54 + coef13*coef25*coef32*coef54 + coef15*coef22*coef33*coef54 - coef12*coef25*coef33*coef54 - coef13*coef22*coef35*coef54 + coef12*coef23*coef35*coef54 + coef14*coef23*coef32*coef55 - coef13*coef24*coef32*coef55 - coef14*coef22*coef33*coef55 + coef12*coef24*coef33*coef55 + coef13*coef22*coef34*coef55 - coef12*coef23*coef34*coef55)/D
    a15 = ( coef15*coef24*coef33*coef42 - coef14*coef25*coef33*coef42 - coef15*coef23*coef34*coef42 + coef13*coef25*coef34*coef42 + coef14*coef23*coef35*coef42 - coef13*coef24*coef35*coef42 - coef15*coef24*coef32*coef43 + coef14*coef25*coef32*coef43 + coef15*coef22*coef34*coef43 - coef12*coef25*coef34*coef43 - coef14*coef22*coef35*coef43 + coef12*coef24*coef35*coef43 + coef15*coef23*coef32*coef44 - coef13*coef25*coef32*coef44 - coef15*coef22*coef33*coef44 + coef12*coef25*coef33*coef44 + coef13*coef22*coef35*coef44 - coef12*coef23*coef35*coef44 - coef14*coef23*coef32*coef45 + coef13*coef24*coef32*coef45 + coef14*coef22*coef33*coef45 - coef12*coef24*coef33*coef45 - coef13*coef22*coef34*coef45 + coef12*coef23*coef34*coef45)/D

    a21 = (-coef25*coef34*coef43*coef51 + coef24*coef35*coef43*coef51 + coef25*coef33*coef44*coef51 - coef23*coef35*coef44*coef51 - coef24*coef33*coef45*coef51 + coef23*coef34*coef45*coef51 + coef25*coef34*coef41*coef53 - coef24*coef35*coef41*coef53 - coef25*coef31*coef44*coef53 + coef21*coef35*coef44*coef53 + coef24*coef31*coef45*coef53 - coef21*coef34*coef45*coef53 - coef25*coef33*coef41*coef54 + coef23*coef35*coef41*coef54 + coef25*coef31*coef43*coef54 - coef21*coef35*coef43*coef54 - coef23*coef31*coef45*coef54 + coef21*coef33*coef45*coef54 + coef24*coef33*coef41*coef55 - coef23*coef34*coef41*coef55 - coef24*coef31*coef43*coef55 + coef21*coef34*coef43*coef55 + coef23*coef31*coef44*coef55 - coef21*coef33*coef44*coef55)/D
    a22 = ( coef15*coef34*coef43*coef51 - coef14*coef35*coef43*coef51 - coef15*coef33*coef44*coef51 + coef13*coef35*coef44*coef51 + coef14*coef33*coef45*coef51 - coef13*coef34*coef45*coef51 - coef15*coef34*coef41*coef53 + coef14*coef35*coef41*coef53 + coef15*coef31*coef44*coef53 - coef11*coef35*coef44*coef53 - coef14*coef31*coef45*coef53 + coef11*coef34*coef45*coef53 + coef15*coef33*coef41*coef54 - coef13*coef35*coef41*coef54 - coef15*coef31*coef43*coef54 + coef11*coef35*coef43*coef54 + coef13*coef31*coef45*coef54 - coef11*coef33*coef45*coef54 - coef14*coef33*coef41*coef55 + coef13*coef34*coef41*coef55 + coef14*coef31*coef43*coef55 - coef11*coef34*coef43*coef55 - coef13*coef31*coef44*coef55 + coef11*coef33*coef44*coef55)/D
    a23 = (-coef15*coef24*coef43*coef51 + coef14*coef25*coef43*coef51 + coef15*coef23*coef44*coef51 - coef13*coef25*coef44*coef51 - coef14*coef23*coef45*coef51 + coef13*coef24*coef45*coef51 + coef15*coef24*coef41*coef53 - coef14*coef25*coef41*coef53 - coef15*coef21*coef44*coef53 + coef11*coef25*coef44*coef53 + coef14*coef21*coef45*coef53 - coef11*coef24*coef45*coef53 - coef15*coef23*coef41*coef54 + coef13*coef25*coef41*coef54 + coef15*coef21*coef43*coef54 - coef11*coef25*coef43*coef54 - coef13*coef21*coef45*coef54 + coef11*coef23*coef45*coef54 + coef14*coef23*coef41*coef55 - coef13*coef24*coef41*coef55 - coef14*coef21*coef43*coef55 + coef11*coef24*coef43*coef55 + coef13*coef21*coef44*coef55 - coef11*coef23*coef44*coef55)/D
    a24 = ( coef15*coef24*coef33*coef51 - coef14*coef25*coef33*coef51 - coef15*coef23*coef34*coef51 + coef13*coef25*coef34*coef51 + coef14*coef23*coef35*coef51 - coef13*coef24*coef35*coef51 - coef15*coef24*coef31*coef53 + coef14*coef25*coef31*coef53 + coef15*coef21*coef34*coef53 - coef11*coef25*coef34*coef53 - coef14*coef21*coef35*coef53 + coef11*coef24*coef35*coef53 + coef15*coef23*coef31*coef54 - coef13*coef25*coef31*coef54 - coef15*coef21*coef33*coef54 + coef11*coef25*coef33*coef54 + coef13*coef21*coef35*coef54 - coef11*coef23*coef35*coef54 - coef14*coef23*coef31*coef55 + coef13*coef24*coef31*coef55 + coef14*coef21*coef33*coef55 - coef11*coef24*coef33*coef55 - coef13*coef21*coef34*coef55 + coef11*coef23*coef34*coef55)/D
    a25 = (-coef15*coef24*coef33*coef41 + coef14*coef25*coef33*coef41 + coef15*coef23*coef34*coef41 - coef13*coef25*coef34*coef41 - coef14*coef23*coef35*coef41 + coef13*coef24*coef35*coef41 + coef15*coef24*coef31*coef43 - coef14*coef25*coef31*coef43 - coef15*coef21*coef34*coef43 + coef11*coef25*coef34*coef43 + coef14*coef21*coef35*coef43 - coef11*coef24*coef35*coef43 - coef15*coef23*coef31*coef44 + coef13*coef25*coef31*coef44 + coef15*coef21*coef33*coef44 - coef11*coef25*coef33*coef44 - coef13*coef21*coef35*coef44 + coef11*coef23*coef35*coef44 + coef14*coef23*coef31*coef45 - coef13*coef24*coef31*coef45 - coef14*coef21*coef33*coef45 + coef11*coef24*coef33*coef45 + coef13*coef21*coef34*coef45 - coef11*coef23*coef34*coef45)/D

    a31 = ( coef25*coef34*coef42*coef51 - coef24*coef35*coef42*coef51 - coef25*coef32*coef44*coef51 + coef22*coef35*coef44*coef51 + coef24*coef32*coef45*coef51 - coef22*coef34*coef45*coef51 - coef25*coef34*coef41*coef52 + coef24*coef35*coef41*coef52 + coef25*coef31*coef44*coef52 - coef21*coef35*coef44*coef52 - coef24*coef31*coef45*coef52 + coef21*coef34*coef45*coef52 + coef25*coef32*coef41*coef54 - coef22*coef35*coef41*coef54 - coef25*coef31*coef42*coef54 + coef21*coef35*coef42*coef54 + coef22*coef31*coef45*coef54 - coef21*coef32*coef45*coef54 - coef24*coef32*coef41*coef55 + coef22*coef34*coef41*coef55 + coef24*coef31*coef42*coef55 - coef21*coef34*coef42*coef55 - coef22*coef31*coef44*coef55 + coef21*coef32*coef44*coef55)/D
    a32 = (-coef15*coef34*coef42*coef51 + coef14*coef35*coef42*coef51 + coef15*coef32*coef44*coef51 - coef12*coef35*coef44*coef51 - coef14*coef32*coef45*coef51 + coef12*coef34*coef45*coef51 + coef15*coef34*coef41*coef52 - coef14*coef35*coef41*coef52 - coef15*coef31*coef44*coef52 + coef11*coef35*coef44*coef52 + coef14*coef31*coef45*coef52 - coef11*coef34*coef45*coef52 - coef15*coef32*coef41*coef54 + coef12*coef35*coef41*coef54 + coef15*coef31*coef42*coef54 - coef11*coef35*coef42*coef54 - coef12*coef31*coef45*coef54 + coef11*coef32*coef45*coef54 + coef14*coef32*coef41*coef55 - coef12*coef34*coef41*coef55 - coef14*coef31*coef42*coef55 + coef11*coef34*coef42*coef55 + coef12*coef31*coef44*coef55 - coef11*coef32*coef44*coef55)/D
    a33 = ( coef15*coef24*coef42*coef51 - coef14*coef25*coef42*coef51 - coef15*coef22*coef44*coef51 + coef12*coef25*coef44*coef51 + coef14*coef22*coef45*coef51 - coef12*coef24*coef45*coef51 - coef15*coef24*coef41*coef52 + coef14*coef25*coef41*coef52 + coef15*coef21*coef44*coef52 - coef11*coef25*coef44*coef52 - coef14*coef21*coef45*coef52 + coef11*coef24*coef45*coef52 + coef15*coef22*coef41*coef54 - coef12*coef25*coef41*coef54 - coef15*coef21*coef42*coef54 + coef11*coef25*coef42*coef54 + coef12*coef21*coef45*coef54 - coef11*coef22*coef45*coef54 - coef14*coef22*coef41*coef55 + coef12*coef24*coef41*coef55 + coef14*coef21*coef42*coef55 - coef11*coef24*coef42*coef55 - coef12*coef21*coef44*coef55 + coef11*coef22*coef44*coef55)/D
    a34 = (-coef15*coef24*coef32*coef51 + coef14*coef25*coef32*coef51 + coef15*coef22*coef34*coef51 - coef12*coef25*coef34*coef51 - coef14*coef22*coef35*coef51 + coef12*coef24*coef35*coef51 + coef15*coef24*coef31*coef52 - coef14*coef25*coef31*coef52 - coef15*coef21*coef34*coef52 + coef11*coef25*coef34*coef52 + coef14*coef21*coef35*coef52 - coef11*coef24*coef35*coef52 - coef15*coef22*coef31*coef54 + coef12*coef25*coef31*coef54 + coef15*coef21*coef32*coef54 - coef11*coef25*coef32*coef54 - coef12*coef21*coef35*coef54 + coef11*coef22*coef35*coef54 + coef14*coef22*coef31*coef55 - coef12*coef24*coef31*coef55 - coef14*coef21*coef32*coef55 + coef11*coef24*coef32*coef55 + coef12*coef21*coef34*coef55 - coef11*coef22*coef34*coef55)/D
    a35 = ( coef15*coef24*coef32*coef41 - coef14*coef25*coef32*coef41 - coef15*coef22*coef34*coef41 + coef12*coef25*coef34*coef41 + coef14*coef22*coef35*coef41 - coef12*coef24*coef35*coef41 - coef15*coef24*coef31*coef42 + coef14*coef25*coef31*coef42 + coef15*coef21*coef34*coef42 - coef11*coef25*coef34*coef42 - coef14*coef21*coef35*coef42 + coef11*coef24*coef35*coef42 + coef15*coef22*coef31*coef44 - coef12*coef25*coef31*coef44 - coef15*coef21*coef32*coef44 + coef11*coef25*coef32*coef44 + coef12*coef21*coef35*coef44 - coef11*coef22*coef35*coef44 - coef14*coef22*coef31*coef45 + coef12*coef24*coef31*coef45 + coef14*coef21*coef32*coef45 - coef11*coef24*coef32*coef45 - coef12*coef21*coef34*coef45 + coef11*coef22*coef34*coef45)/D
    
    a41 = (-coef25*coef33*coef42*coef51 + coef23*coef35*coef42*coef51 + coef25*coef32*coef43*coef51 - coef22*coef35*coef43*coef51 - coef23*coef32*coef45*coef51 + coef22*coef33*coef45*coef51 + coef25*coef33*coef41*coef52 - coef23*coef35*coef41*coef52 - coef25*coef31*coef43*coef52 + coef21*coef35*coef43*coef52 + coef23*coef31*coef45*coef52 - coef21*coef33*coef45*coef52 - coef25*coef32*coef41*coef53 + coef22*coef35*coef41*coef53 + coef25*coef31*coef42*coef53 - coef21*coef35*coef42*coef53 - coef22*coef31*coef45*coef53 + coef21*coef32*coef45*coef53 + coef23*coef32*coef41*coef55 - coef22*coef33*coef41*coef55 - coef23*coef31*coef42*coef55 + coef21*coef33*coef42*coef55 + coef22*coef31*coef43*coef55 - coef21*coef32*coef43*coef55)/D
    a42 = ( coef15*coef33*coef42*coef51 - coef13*coef35*coef42*coef51 - coef15*coef32*coef43*coef51 + coef12*coef35*coef43*coef51 + coef13*coef32*coef45*coef51 - coef12*coef33*coef45*coef51 - coef15*coef33*coef41*coef52 + coef13*coef35*coef41*coef52 + coef15*coef31*coef43*coef52 - coef11*coef35*coef43*coef52 - coef13*coef31*coef45*coef52 + coef11*coef33*coef45*coef52 + coef15*coef32*coef41*coef53 - coef12*coef35*coef41*coef53 - coef15*coef31*coef42*coef53 + coef11*coef35*coef42*coef53 + coef12*coef31*coef45*coef53 - coef11*coef32*coef45*coef53 - coef13*coef32*coef41*coef55 + coef12*coef33*coef41*coef55 + coef13*coef31*coef42*coef55 - coef11*coef33*coef42*coef55 - coef12*coef31*coef43*coef55 + coef11*coef32*coef43*coef55)/D
    a43 = (-coef15*coef23*coef42*coef51 + coef13*coef25*coef42*coef51 + coef15*coef22*coef43*coef51 - coef12*coef25*coef43*coef51 - coef13*coef22*coef45*coef51 + coef12*coef23*coef45*coef51 + coef15*coef23*coef41*coef52 - coef13*coef25*coef41*coef52 - coef15*coef21*coef43*coef52 + coef11*coef25*coef43*coef52 + coef13*coef21*coef45*coef52 - coef11*coef23*coef45*coef52 - coef15*coef22*coef41*coef53 + coef12*coef25*coef41*coef53 + coef15*coef21*coef42*coef53 - coef11*coef25*coef42*coef53 - coef12*coef21*coef45*coef53 + coef11*coef22*coef45*coef53 + coef13*coef22*coef41*coef55 - coef12*coef23*coef41*coef55 - coef13*coef21*coef42*coef55 + coef11*coef23*coef42*coef55 + coef12*coef21*coef43*coef55 - coef11*coef22*coef43*coef55)/D
    a44 = ( coef15*coef23*coef32*coef51 - coef13*coef25*coef32*coef51 - coef15*coef22*coef33*coef51 + coef12*coef25*coef33*coef51 + coef13*coef22*coef35*coef51 - coef12*coef23*coef35*coef51 - coef15*coef23*coef31*coef52 + coef13*coef25*coef31*coef52 + coef15*coef21*coef33*coef52 - coef11*coef25*coef33*coef52 - coef13*coef21*coef35*coef52 + coef11*coef23*coef35*coef52 + coef15*coef22*coef31*coef53 - coef12*coef25*coef31*coef53 - coef15*coef21*coef32*coef53 + coef11*coef25*coef32*coef53 + coef12*coef21*coef35*coef53 - coef11*coef22*coef35*coef53 - coef13*coef22*coef31*coef55 + coef12*coef23*coef31*coef55 + coef13*coef21*coef32*coef55 - coef11*coef23*coef32*coef55 - coef12*coef21*coef33*coef55 + coef11*coef22*coef33*coef55)/D
    a45 = (-coef15*coef23*coef32*coef41 + coef13*coef25*coef32*coef41 + coef15*coef22*coef33*coef41 - coef12*coef25*coef33*coef41 - coef13*coef22*coef35*coef41 + coef12*coef23*coef35*coef41 + coef15*coef23*coef31*coef42 - coef13*coef25*coef31*coef42 - coef15*coef21*coef33*coef42 + coef11*coef25*coef33*coef42 + coef13*coef21*coef35*coef42 - coef11*coef23*coef35*coef42 - coef15*coef22*coef31*coef43 + coef12*coef25*coef31*coef43 + coef15*coef21*coef32*coef43 - coef11*coef25*coef32*coef43 - coef12*coef21*coef35*coef43 + coef11*coef22*coef35*coef43 + coef13*coef22*coef31*coef45 - coef12*coef23*coef31*coef45 - coef13*coef21*coef32*coef45 + coef11*coef23*coef32*coef45 + coef12*coef21*coef33*coef45 - coef11*coef22*coef33*coef45)/D
    
    a51 = ( coef24*coef33*coef42*coef51 - coef23*coef34*coef42*coef51 - coef24*coef32*coef43*coef51 + coef22*coef34*coef43*coef51 + coef23*coef32*coef44*coef51 - coef22*coef33*coef44*coef51 - coef24*coef33*coef41*coef52 + coef23*coef34*coef41*coef52 + coef24*coef31*coef43*coef52 - coef21*coef34*coef43*coef52 - coef23*coef31*coef44*coef52 + coef21*coef33*coef44*coef52 + coef24*coef32*coef41*coef53 - coef22*coef34*coef41*coef53 - coef24*coef31*coef42*coef53 + coef21*coef34*coef42*coef53 + coef22*coef31*coef44*coef53 - coef21*coef32*coef44*coef53 - coef23*coef32*coef41*coef54 + coef22*coef33*coef41*coef54 + coef23*coef31*coef42*coef54 - coef21*coef33*coef42*coef54 - coef22*coef31*coef43*coef54 + coef21*coef32*coef43*coef54)/D
    a52 = (-coef14*coef33*coef42*coef51 + coef13*coef34*coef42*coef51 + coef14*coef32*coef43*coef51 - coef12*coef34*coef43*coef51 - coef13*coef32*coef44*coef51 + coef12*coef33*coef44*coef51 + coef14*coef33*coef41*coef52 - coef13*coef34*coef41*coef52 - coef14*coef31*coef43*coef52 + coef11*coef34*coef43*coef52 + coef13*coef31*coef44*coef52 - coef11*coef33*coef44*coef52 - coef14*coef32*coef41*coef53 + coef12*coef34*coef41*coef53 + coef14*coef31*coef42*coef53 - coef11*coef34*coef42*coef53 - coef12*coef31*coef44*coef53 + coef11*coef32*coef44*coef53 + coef13*coef32*coef41*coef54 - coef12*coef33*coef41*coef54 - coef13*coef31*coef42*coef54 + coef11*coef33*coef42*coef54 + coef12*coef31*coef43*coef54 - coef11*coef32*coef43*coef54)/D
    a53 = ( coef14*coef23*coef42*coef51 - coef13*coef24*coef42*coef51 - coef14*coef22*coef43*coef51 + coef12*coef24*coef43*coef51 + coef13*coef22*coef44*coef51 - coef12*coef23*coef44*coef51 - coef14*coef23*coef41*coef52 + coef13*coef24*coef41*coef52 + coef14*coef21*coef43*coef52 - coef11*coef24*coef43*coef52 - coef13*coef21*coef44*coef52 + coef11*coef23*coef44*coef52 + coef14*coef22*coef41*coef53 - coef12*coef24*coef41*coef53 - coef14*coef21*coef42*coef53 + coef11*coef24*coef42*coef53 + coef12*coef21*coef44*coef53 - coef11*coef22*coef44*coef53 - coef13*coef22*coef41*coef54 + coef12*coef23*coef41*coef54 + coef13*coef21*coef42*coef54 - coef11*coef23*coef42*coef54 - coef12*coef21*coef43*coef54 + coef11*coef22*coef43*coef54)/D
    a54 = (-coef14*coef23*coef32*coef51 + coef13*coef24*coef32*coef51 + coef14*coef22*coef33*coef51 - coef12*coef24*coef33*coef51 - coef13*coef22*coef34*coef51 + coef12*coef23*coef34*coef51 + coef14*coef23*coef31*coef52 - coef13*coef24*coef31*coef52 - coef14*coef21*coef33*coef52 + coef11*coef24*coef33*coef52 + coef13*coef21*coef34*coef52 - coef11*coef23*coef34*coef52 - coef14*coef22*coef31*coef53 + coef12*coef24*coef31*coef53 + coef14*coef21*coef32*coef53 - coef11*coef24*coef32*coef53 - coef12*coef21*coef34*coef53 + coef11*coef22*coef34*coef53 + coef13*coef22*coef31*coef54 - coef12*coef23*coef31*coef54 - coef13*coef21*coef32*coef54 + coef11*coef23*coef32*coef54 + coef12*coef21*coef33*coef54 - coef11*coef22*coef33*coef54)/D
    a55 = ( coef14*coef23*coef32*coef41 - coef13*coef24*coef32*coef41 - coef14*coef22*coef33*coef41 + coef12*coef24*coef33*coef41 + coef13*coef22*coef34*coef41 - coef12*coef23*coef34*coef41 - coef14*coef23*coef31*coef42 + coef13*coef24*coef31*coef42 + coef14*coef21*coef33*coef42 - coef11*coef24*coef33*coef42 - coef13*coef21*coef34*coef42 + coef11*coef23*coef34*coef42 + coef14*coef22*coef31*coef43 - coef12*coef24*coef31*coef43 - coef14*coef21*coef32*coef43 + coef11*coef24*coef32*coef43 + coef12*coef21*coef34*coef43 - coef11*coef22*coef34*coef43 - coef13*coef22*coef31*coef44 + coef12*coef23*coef31*coef44 + coef13*coef21*coef32*coef44 - coef11*coef23*coef32*coef44 - coef12*coef21*coef33*coef44 + coef11*coef22*coef33*coef44)/D

    return a11,a12,a13,a14,a15,a21,a22,a23,a24,a25,a31,a32,a33,a34,a35,a41,a42,a43,a44,a45,a51,a52,a53,a54,a55,bool_aux

def bathymetry_mid_point2(mesh):

    Xj = mesh["Bj"]
    Lx = mesh["Bx"]
    Ly = mesh["By"]
    xj = mesh["xj"]
    yj = mesh["yj"]

    jk = mesh["jk"]
    ijk = mesh["Neighboors"]
    choice = mesh["cChoose1"]

    xjk=mesh["Coordinates"][0,mesh["Elements"]]
    yjk=mesh["Coordinates"][1,mesh["Elements"]]

    c = 0.5+np.sqrt(3)/6

    xmjk1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    xmjk2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]

    ymjk1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    ymjk2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]

    Xj121 = Xj + Lx*(xmjk1[0,:]-xj) + Ly*(ymjk1[0,:]-yj)
    Xj122 = Xj + Lx*(xmjk2[0,:]-xj) + Ly*(ymjk2[0,:]-yj)

    Xj231 = Xj + Lx*(xmjk1[1,:]-xj) + Ly*(ymjk1[1,:]-yj)
    Xj232 = Xj + Lx*(xmjk2[1,:]-xj) + Ly*(ymjk2[1,:]-yj)
    
    Xj311 = Xj + Lx*(xmjk1[2,:]-xj) + Ly*(ymjk1[2,:]-yj)
    Xj312 = Xj + Lx*(xmjk2[2,:]-xj) + Ly*(ymjk2[2,:]-yj)

    Bml1 = np.array([Xj121, Xj231, Xj311])
    Bml2 = np.array([Xj122, Xj232, Xj312])

    Bmr1jk = Bml1.take(jk)
    Bmr2jk = Bml2.take(jk)
    Bmr1 = np.where(choice==0, Bmr1jk, Bmr2jk)
    Bmr2 = np.where(choice==1, Bmr1jk, Bmr2jk)

    #Well-balanced reconstruction
    Bmj1 = np.maximum(Bml1, Bmr1)
    Bmj2 = np.maximum(Bml2, Bmr2)

    #Fix cells that have no neighbors
    Bmj1[ijk < 0] = Bml1[ijk < 0]
    Bmj2[ijk < 0] = Bml2[ijk < 0]

    mesh["Bmj1"] = Bmj1
    mesh["Bmj2"] = Bmj2

    return

def bathymetry_mid_point22(mesh):

    Bpoints = mesh["Bpoints"]
    xj = mesh["xj"]
    yj = mesh["yj"]

    jk = mesh["jk"]
    ijk = mesh["Neighboors"]
    choice = mesh["cChoose1"]

    xjk=mesh["Coordinates"][0,mesh["Elements"]]
    yjk=mesh["Coordinates"][1,mesh["Elements"]]

    Bpol = CloughTocher2DInterpolator(mesh["Coordinates"].T,Bpoints,np.min(Bpoints),rescale=True)

    c = 0.5+np.sqrt(3)/6

    xmjk1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    xmjk2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]

    ymjk1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    ymjk2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]

    Xj121 = Bpol(xmjk1[0,:],ymjk1[0,:])
    Xj122 = Bpol(xmjk2[0,:],ymjk2[0,:])

    Xj231 = Bpol(xmjk1[1,:],ymjk1[1,:])
    Xj232 = Bpol(xmjk2[1,:],ymjk2[1,:])
    
    Xj311 = Bpol(xmjk1[2,:],ymjk1[2,:])
    Xj312 = Bpol(xmjk2[2,:],ymjk2[2,:])

    Bml1 = np.array([Xj121, Xj231, Xj311])
    Bml2 = np.array([Xj122, Xj232, Xj312])

    Bmr1jk = Bml1.take(jk)
    Bmr2jk = Bml2.take(jk)
    Bmr1 = np.where(choice==0, Bmr1jk, Bmr2jk)
    Bmr2 = np.where(choice==1, Bmr1jk, Bmr2jk)

    #Well-balanced reconstruction
    Bmj1 = np.maximum(Bml1, Bmr1)
    Bmj2 = np.maximum(Bml2, Bmr2)

    #Fix cells that have no neighbors
    Bmj1[ijk < 0] = Bml1[ijk < 0]
    Bmj2[ijk < 0] = Bml2[ijk < 0]

    mesh["Bmj1"] = Bmj1
    mesh["Bmj2"] = Bmj2

    mesh["Bpol"] = Bpol

    return

def bathymetry_mid_point23(mesh):

    jk = mesh["jk"]
    ijk = mesh["Neighboors"]
    choice = mesh["cChoose1"]
    ov = mesh["no_2nd_neigh"] | mesh["no_1st_neigh"]

    xjk=mesh["Coordinates"][0,mesh["Elements"]]
    yjk=mesh["Coordinates"][1,mesh["Elements"]]

    Bpol, Bx, By =PieceWiseReconstruction.weno2(mesh["Bj"], mesh["Tj"], mesh["Ix"], mesh["Iy"], mesh["Ixy"], mesh["xj"], mesh["yj"],xjk,yjk, mesh["lsq_coefs"], mesh["lin_coefs2"], mesh["Neighboors"], mesh["Neighs2"],ov,mesh["Constants"]["Tolerance"])

    c = 0.5+np.sqrt(3)/6

    xmjk1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    xmjk2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]

    ymjk1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    ymjk2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]

    Xj121 = Bpol(xmjk1[0,:],ymjk1[0,:],ov)
    Xj122 = Bpol(xmjk2[0,:],ymjk2[0,:],ov)

    Xj231 = Bpol(xmjk1[1,:],ymjk1[1,:],ov)
    Xj232 = Bpol(xmjk2[1,:],ymjk2[1,:],ov)
    
    Xj311 = Bpol(xmjk1[2,:],ymjk1[2,:],ov)
    Xj312 = Bpol(xmjk2[2,:],ymjk2[2,:],ov)

    Bml1 = np.array([Xj121, Xj231, Xj311])
    Bml2 = np.array([Xj122, Xj232, Xj312])

    Bmr1jk = Bml1.take(jk)
    Bmr2jk = Bml2.take(jk)
    Bmr1 = np.where(choice==0, Bmr1jk, Bmr2jk)
    Bmr2 = np.where(choice==1, Bmr1jk, Bmr2jk)

    #Well-balanced reconstruction
    Bmj1 = np.maximum(Bml1, Bmr1)
    Bmj2 = np.maximum(Bml2, Bmr2)

    #Fix cells that have no neighbors
    Bmj1[ijk < 0] = Bml1[ijk < 0]
    Bmj2[ijk < 0] = Bml2[ijk < 0]

    mesh["Bmj1"] = Bmj1
    mesh["Bmj2"] = Bmj2

    mesh["Bpol"] = Bpol

    return

def in_point3(mesh):
    xjk=mesh["Coordinates"][0,mesh["Elements"]]
    yjk=mesh["Coordinates"][1,mesh["Elements"]]

    xj = mesh["xj"]
    yj = mesh["yj"]

    xg0 = (xjk[0,:]+xj)/2
    yg0 = (yjk[0,:]+yj)/2

    xg1 = (xjk[1,:]+xj)/2
    yg1 = (yjk[1,:]+yj)/2

    xg2 = (xjk[2,:]+xj)/2
    yg2 = (yjk[2,:]+yj)/2

    mesh["xg0"]=xg0
    mesh["xg1"]=xg1
    mesh["xg2"]=xg2

    mesh["yg0"]=yg0
    mesh["yg1"]=yg1
    mesh["yg2"]=yg2

    return

def bathymetry_in_point3(mesh):

    Xj = mesh["Bj"]
    Lx = mesh["Bx"]
    Ly = mesh["By"]
    xj = mesh["xj"]
    yj = mesh["yj"]

    jk = mesh["jk"]
    ijk = mesh["Neighboors"]

    xg0=mesh["xg0"]
    xg1=mesh["xg1"]
    xg2=mesh["xg2"]

    yg0=mesh["yg0"]
    yg1=mesh["yg1"]
    yg2=mesh["yg2"]

    Bg0 = Xj + Lx*(xg0-xj) + Ly*(yg0-yj)

    Bg1 = Xj + Lx*(xg1-xj) + Ly*(yg1-yj)
    
    Bg2 = Xj + Lx*(xg2-xj) + Ly*(yg2-yj)

    mesh["Bg0"] = Bg0
    mesh["Bg1"] = Bg1
    mesh["Bg2"] = Bg2

    return

def bathymetry_in_point32(mesh):

    Bpol = mesh["Bpol"]

    xg0=mesh["xg0"]
    xg1=mesh["xg1"]
    xg2=mesh["xg2"]

    yg0=mesh["yg0"]
    yg1=mesh["yg1"]
    yg2=mesh["yg2"]

    Bg0 = Bpol(xg0,yg0)

    Bg1 = Bpol(xg1,yg1)
    
    Bg2 = Bpol(xg2,yg2)

    mesh["Bg0"] = Bg0
    mesh["Bg1"] = Bg1
    mesh["Bg2"] = Bg2

    return

def bathymetry_in_point33(mesh):

    Bpol = mesh["Bpol"]
    
    ov = mesh["no_2nd_neigh"] | mesh["no_1st_neigh"]

    xg0=mesh["xg0"]
    xg1=mesh["xg1"]
    xg2=mesh["xg2"]

    yg0=mesh["yg0"]
    yg1=mesh["yg1"]
    yg2=mesh["yg2"]

    Bg0 = Bpol(xg0,yg0,ov)

    Bg1 = Bpol(xg1,yg1,ov)
    
    Bg2 = Bpol(xg2,yg2,ov)

    mesh["Bg0"] = Bg0
    mesh["Bg1"] = Bg1
    mesh["Bg2"] = Bg2

    return

def inner_outer_midpoint(mesh):

    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]
    yjk         = Coordinates[1,Elements]
    jk          = mesh["jk"]

    c = 0.5+np.sqrt(3)/6

    xin1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    xin2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]
    yin1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    yin2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]

    deltax11=xin1-xin1.take(jk)
    deltax12=xin1-xin2.take(jk)

    deltay11=yin1-yin1.take(jk)
    deltay12=yin1-yin2.take(jk)

    #deltax21=xin2-xin1.take(jk)
    #deltax22=xin2-xin2.take(jk)

    #deltay21=yin2-yin1.take(jk)
    #deltay22=yin2-yin2.take(jk)

    choice11 = np.where((deltax11 == 0)&(deltay11 == 0),0,1)
    choice12 = np.where((deltax12 == 0)&(deltay12 == 0),1,0)
    
    #choice21 = np.where((deltax21 == 0)&(deltay21 == 0),0,1)
    #choice22 = np.where((deltax22 == 0)&(deltay22 == 0),1,0)

    #print((choice11==choice12).all())
    #print((choice21==choice22).all())

    mesh["cChoose1"]=choice11
    #mesh["cChoose2"]=choice21

    return

def precomp_minmod2(mesh):

    neighbor2(mesh)
    second_moments(mesh)
    in_point3(mesh)
    bathymetry_mid_point2(mesh)
    bathymetry_in_point3(mesh)
    stencils1,stencils2=create_stencils()
    
    mesh["NChoose"]=(np.dstack((mesh["Neighboors"].T,mesh["Neighboors"].T,mesh["Neighboors"].T)),mesh["Neighs2"])

    mesh["Stencils"]={}
    zeroes=np.zeros_like(mesh["xj"])

    mesh["Stencils"]["aij"]=np.tile(zeroes,(12,5,5,1))
    mesh["Stencils"]["n2n"]=[0,0,0,0,0,0,0,0,0,0,0,0]
    mesh["Stencils"]["bool_aux"]=[0,0,0,0,0,0,0,0,0,0,0,0]

    i=0
    for stencil in stencils2:
        mesh["Stencils"]["n2n"][i]=stencil

        coef11,coef12,coef13,coef14,coef15,coef21,coef22,coef23,coef24,coef25,coef31,coef32,coef33,coef34,coef35,coef41,coef42,coef43,coef44,coef45,coef51,coef52,coef53,coef54,coef55 = quad_coefs(mesh,stencil)
        a11,a12,a13,a14,a15,a21,a22,a23,a24,a25,a31,a32,a33,a34,a35,a41,a42,a43,a44,a45,a51,a52,a53,a54,a55,bool_aux = inverse_coefs(coef11,coef12,coef13,coef14,coef15,coef21,coef22,coef23,coef24,coef25,coef31,coef32,coef33,coef34,coef35,coef41,coef42,coef43,coef44,coef45,coef51,coef52,coef53,coef54,coef55)

        mesh["Stencils"]["bool_aux"][i]=bool_aux

        mesh["Stencils"]["aij"][i][0,0]=a11
        mesh["Stencils"]["aij"][i][0,1]=a12
        mesh["Stencils"]["aij"][i][0,2]=a13
        mesh["Stencils"]["aij"][i][0,3]=a14
        mesh["Stencils"]["aij"][i][0,4]=a15
        mesh["Stencils"]["aij"][i][1,0]=a21
        mesh["Stencils"]["aij"][i][1,1]=a22
        mesh["Stencils"]["aij"][i][1,2]=a23
        mesh["Stencils"]["aij"][i][1,3]=a24
        mesh["Stencils"]["aij"][i][1,4]=a25
        mesh["Stencils"]["aij"][i][2,0]=a31
        mesh["Stencils"]["aij"][i][2,1]=a32
        mesh["Stencils"]["aij"][i][2,2]=a33
        mesh["Stencils"]["aij"][i][2,3]=a34
        mesh["Stencils"]["aij"][i][2,4]=a35
        mesh["Stencils"]["aij"][i][3,0]=a41
        mesh["Stencils"]["aij"][i][3,1]=a42
        mesh["Stencils"]["aij"][i][3,2]=a43
        mesh["Stencils"]["aij"][i][3,3]=a44
        mesh["Stencils"]["aij"][i][3,4]=a45
        mesh["Stencils"]["aij"][i][4,0]=a51
        mesh["Stencils"]["aij"][i][4,1]=a52
        mesh["Stencils"]["aij"][i][4,2]=a53
        mesh["Stencils"]["aij"][i][4,3]=a54
        mesh["Stencils"]["aij"][i][4,4]=a55

        i+=1
    
    return

def lsq_coefs(mesh):

    xj0 = mesh["xj"]
    xj1 = mesh["xj"][mesh["Neighboors"][0,:]]
    xj2 = mesh["xj"][mesh["Neighboors"][1,:]]
    xj3 = mesh["xj"][mesh["Neighboors"][2,:]]
    xj4 = mesh["xj"][mesh["Neighs2"][:,0,0]]
    xj5 = mesh["xj"][mesh["Neighs2"][:,0,1]]
    xj6 = mesh["xj"][mesh["Neighs2"][:,1,0]]
    xj7 = mesh["xj"][mesh["Neighs2"][:,1,1]]
    xj8 = mesh["xj"][mesh["Neighs2"][:,2,0]]
    xj9 = mesh["xj"][mesh["Neighs2"][:,2,1]]
    
    yj0 = mesh["yj"]
    yj1 = mesh["yj"][mesh["Neighboors"][0,:]]
    yj2 = mesh["yj"][mesh["Neighboors"][1,:]]
    yj3 = mesh["yj"][mesh["Neighboors"][2,:]]
    yj4 = mesh["yj"][mesh["Neighs2"][:,0,0]]
    yj5 = mesh["yj"][mesh["Neighs2"][:,0,1]]
    yj6 = mesh["yj"][mesh["Neighs2"][:,1,0]]
    yj7 = mesh["yj"][mesh["Neighs2"][:,1,1]]
    yj8 = mesh["yj"][mesh["Neighs2"][:,2,0]]
    yj9 = mesh["yj"][mesh["Neighs2"][:,2,1]]

    Tj0 = mesh["Tj"]
    Tj1 = mesh["Tj"][mesh["Neighboors"][0,:]]
    Tj2 = mesh["Tj"][mesh["Neighboors"][1,:]]
    Tj3 = mesh["Tj"][mesh["Neighboors"][2,:]]
    Tj4 = mesh["Tj"][mesh["Neighs2"][:,0,0]]
    Tj5 = mesh["Tj"][mesh["Neighs2"][:,0,1]]
    Tj6 = mesh["Tj"][mesh["Neighs2"][:,1,0]]
    Tj7 = mesh["Tj"][mesh["Neighs2"][:,1,1]]
    Tj8 = mesh["Tj"][mesh["Neighs2"][:,2,0]]
    Tj9 = mesh["Tj"][mesh["Neighs2"][:,2,1]]


    Iyj0 = mesh["Iy"]
    Iyj1 = mesh["Iy"][mesh["Neighboors"][0,:]]
    Iyj2 = mesh["Iy"][mesh["Neighboors"][1,:]]
    Iyj3 = mesh["Iy"][mesh["Neighboors"][2,:]]
    Iyj4 = mesh["Iy"][mesh["Neighs2"][:,0,0]]
    Iyj5 = mesh["Iy"][mesh["Neighs2"][:,0,1]]
    Iyj6 = mesh["Iy"][mesh["Neighs2"][:,1,0]]
    Iyj7 = mesh["Iy"][mesh["Neighs2"][:,1,1]]
    Iyj8 = mesh["Iy"][mesh["Neighs2"][:,2,0]]
    Iyj9 = mesh["Iy"][mesh["Neighs2"][:,2,1]]

    Ixj0 = mesh["Ix"]
    Ixj1 = mesh["Ix"][mesh["Neighboors"][0,:]]
    Ixj2 = mesh["Ix"][mesh["Neighboors"][1,:]]
    Ixj3 = mesh["Ix"][mesh["Neighboors"][2,:]]
    Ixj4 = mesh["Ix"][mesh["Neighs2"][:,0,0]]
    Ixj5 = mesh["Ix"][mesh["Neighs2"][:,0,1]]
    Ixj6 = mesh["Ix"][mesh["Neighs2"][:,1,0]]
    Ixj7 = mesh["Ix"][mesh["Neighs2"][:,1,1]]
    Ixj8 = mesh["Ix"][mesh["Neighs2"][:,2,0]]
    Ixj9 = mesh["Ix"][mesh["Neighs2"][:,2,1]]

    Ixyj0 = mesh["Ixy"]
    Ixyj1 = mesh["Ixy"][mesh["Neighboors"][0,:]]
    Ixyj2 = mesh["Ixy"][mesh["Neighboors"][1,:]]
    Ixyj3 = mesh["Ixy"][mesh["Neighboors"][2,:]]
    Ixyj4 = mesh["Ixy"][mesh["Neighs2"][:,0,0]]
    Ixyj5 = mesh["Ixy"][mesh["Neighs2"][:,0,1]]
    Ixyj6 = mesh["Ixy"][mesh["Neighs2"][:,1,0]]
    Ixyj7 = mesh["Ixy"][mesh["Neighs2"][:,1,1]]
    Ixyj8 = mesh["Ixy"][mesh["Neighs2"][:,2,0]]
    Ixyj9 = mesh["Ixy"][mesh["Neighs2"][:,2,1]]

    #ata11 = np.zeros_like(xj0)+10
    #ata12 = xj0 + xj1 + xj2 + xj3 + xj4 + xj5 + xj6 + xj7 + xj8 + xj9
    #ata13 = yj0 + yj1 + yj2 + yj3 + yj4 + yj5 + yj6 + yj7 + yj8 + yj9
    #ata14 = Iyj0/Tj0 + Iyj1/Tj1 + Iyj2/Tj2 + Iyj3/Tj3 + Iyj4/Tj4 + Iyj5/Tj5 + Iyj6/Tj6 + Iyj7/Tj7 + Iyj8/Tj8 + Iyj9/Tj9
    #ata15 = Ixj0/Tj0 + Ixj1/Tj1 + Ixj2/Tj2 + Ixj3/Tj3 + Ixj4/Tj4 + Ixj5/Tj5 + Ixj6/Tj6 + Ixj7/Tj7 + Ixj8/Tj8 + Ixj9/Tj9
    #ata16 = Ixyj0/Tj0 + Ixyj1/Tj1 + Ixyj2/Tj2 + Ixyj3/Tj3 + Ixyj4/Tj4 + Ixyj5/Tj5 + Ixyj6/Tj6 + Ixyj7/Tj7 + Ixyj8/Tj8 + Ixyj9/Tj9
#
    #ata22 = xj0**2 + xj1**2 + xj2**2 + xj3**2 + xj4**2 + xj5**2 + xj6**2 + xj7**2 + xj8**2 + xj9**2
    #ata23 = xj0*yj0 + xj1*yj1 + xj2*yj2 + xj3*yj3 + xj4*yj4 + xj5*yj5 + xj6*yj6 + xj7*yj7 + xj8*yj8 + xj9*yj9
    #ata24 = (Iyj0*xj0)/Tj0 + (Iyj1*xj1)/Tj1 + (Iyj2*xj2)/Tj2 + (Iyj3*xj3)/Tj3 + (Iyj4*xj4)/Tj4 + (Iyj5*xj5)/Tj5 + (Iyj6*xj6)/Tj6 + (Iyj7*xj7)/Tj7 + (Iyj8*xj8)/Tj8 + (Iyj9*xj9)/Tj9
    #ata25 = (Ixj0*xj0)/Tj0 + (Ixj1*xj1)/Tj1 + (Ixj2*xj2)/Tj2 + (Ixj3*xj3)/Tj3 + (Ixj4*xj4)/Tj4 + (Ixj5*xj5)/Tj5 + (Ixj6*xj6)/Tj6 + (Ixj7*xj7)/Tj7 + (Ixj8*xj8)/Tj8 + (Ixj9*xj9)/Tj9
    #ata26 = (Ixyj0*xj0)/Tj0 + (Ixyj1*xj1)/Tj1 + (Ixyj2*xj2)/Tj2 + (Ixyj3*xj3)/Tj3 + (Ixyj4*xj4)/Tj4 + (Ixyj5*xj5)/Tj5 + (Ixyj6*xj6)/Tj6 + (Ixyj7*xj7)/Tj7 + (Ixyj8*xj8)/Tj8 + (Ixyj9*xj9)/Tj9
#
    #ata33 = yj0**2 + yj1**2 + yj2**2 + yj3**2 + yj4**2 + yj5**2 + yj6**2 + yj7**2 + yj8**2 + yj9**2
    #ata34 = (Iyj0*yj0)/Tj0 + (Iyj1*yj1)/Tj1 + (Iyj2*yj2)/Tj2 + (Iyj3*yj3)/Tj3 + (Iyj4*yj4)/Tj4 + (Iyj5*yj5)/Tj5 + (Iyj6*yj6)/Tj6 + (Iyj7*yj7)/Tj7 + (Iyj8*yj8)/Tj8 + (Iyj9*yj9)/Tj9
    #ata35 = (Ixj0*yj0)/Tj0 + (Ixj1*yj1)/Tj1 + (Ixj2*yj2)/Tj2 + (Ixj3*yj3)/Tj3 + (Ixj4*yj4)/Tj4 + (Ixj5*yj5)/Tj5 + (Ixj6*yj6)/Tj6 + (Ixj7*yj7)/Tj7 + (Ixj8*yj8)/Tj8 + (Ixj9*yj9)/Tj9
    #ata36 = (Ixyj0*yj0)/Tj0 + (Ixyj1*yj1)/Tj1 + (Ixyj2*yj2)/Tj2 + (Ixyj3*yj3)/Tj3 + (Ixyj4*yj4)/Tj4 + (Ixyj5*yj5)/Tj5 + (Ixyj6*yj6)/Tj6 + (Ixyj7*yj7)/Tj7 + (Ixyj8*yj8)/Tj8 + (Ixyj9*yj9)/Tj9
#
    #ata44 = Iyj0**2/Tj0**2 + Iyj1**2/Tj1**2 + Iyj2**2/Tj2**2 + Iyj3**2/Tj3**2 + Iyj4**2/Tj4**2 + Iyj5**2/Tj5**2 + Iyj6**2/Tj6**2 + Iyj7**2/Tj7**2 + Iyj8**2/Tj8**2 + Iyj9**2/Tj9**2
    #ata45 = (Ixj0*Iyj0)/Tj0**2 + (Ixj1*Iyj1)/Tj1**2 + (Ixj2*Iyj2)/Tj2**2 + (Ixj3*Iyj3)/Tj3**2 + (Ixj4*Iyj4)/Tj4**2 + (Ixj5*Iyj5)/Tj5**2 + (Ixj6*Iyj6)/Tj6**2 + (Ixj7*Iyj7)/Tj7**2 + (Ixj8*Iyj8)/Tj8**2 + (Ixj9*Iyj9)/Tj9**2
    #ata46 = (Ixyj0*Iyj0)/Tj0**2 + (Ixyj1*Iyj1)/Tj1**2 + (Ixyj2*Iyj2)/Tj2**2 + (Ixyj3*Iyj3)/Tj3**2 + (Ixyj4*Iyj4)/Tj4**2 + (Ixyj5*Iyj5)/Tj5**2 + (Ixyj6*Iyj6)/Tj6**2 + (Ixyj7*Iyj7)/Tj7**2 + (Ixyj8*Iyj8)/Tj8**2 + (Ixyj9*Iyj9)/Tj9**2
#
    #ata55 = Ixj0**2/Tj0**2 + Ixj1**2/Tj1**2 + Ixj2**2/Tj2**2 + Ixj3**2/Tj3**2 + Ixj4**2/Tj4**2 + Ixj5**2/Tj5**2 + Ixj6**2/Tj6**2 + Ixj7**2/Tj7**2 + Ixj8**2/Tj8**2 + Ixj9**2/Tj9**2
    #ata56 = (Ixj0*Ixyj0)/Tj0**2 + (Ixj1*Ixyj1)/Tj1**2 + (Ixj2*Ixyj2)/Tj2**2 + (Ixj3*Ixyj3)/Tj3**2 + (Ixj4*Ixyj4)/Tj4**2 + (Ixj5*Ixyj5)/Tj5**2 + (Ixj6*Ixyj6)/Tj6**2 + (Ixj7*Ixyj7)/Tj7**2 + (Ixj8*Ixyj8)/Tj8**2 + (Ixj9*Ixyj9)/Tj9**2
#
    #ata66 = Ixyj0**2/Tj0**2 + Ixyj1**2/Tj1**2 + Ixyj2**2/Tj2**2 + Ixyj3**2/Tj3**2 + Ixyj4**2/Tj4**2 + Ixyj5**2/Tj5**2 + Ixyj6**2/Tj6**2 + Ixyj7**2/Tj7**2 + Ixyj8**2/Tj8**2 + Ixyj9**2/Tj9**2
#
    #ata = np.vstack((np.hstack((ata11[None,None,:], ata12[None,None,:], ata13[None,None,:], ata14[None,None,:], ata15[None,None,:], ata16[None,None,:])), np.hstack((ata12[None,None,:], ata22[None,None,:], ata23[None,None,:], ata24[None,None,:], ata25[None,None,:], ata26[None,None,:])), np.hstack((ata13[None,None,:], ata23[None,None,:], ata33[None,None,:], ata34[None,None,:], ata35[None,None,:], ata36[None,None,:])), np.hstack((ata14[None,None,:], ata24[None,None,:], ata34[None,None,:], ata44[None,None,:], ata45[None,None,:], ata46[None,None,:])), np.hstack((ata15[None,None,:], ata25[None,None,:], ata35[None,None,:], ata45[None,None,:], ata55[None,None,:], ata56[None,None,:])), np.hstack((ata16[None,None,:], ata26[None,None,:], ata36[None,None,:], ata46[None,None,:], ata56[None,None,:], ata66[None,None,:]))))

    a = np.vstack((np.hstack(( xj1[None,None,:]-xj0[None,None,:], yj1[None,None,:]-yj0[None,None,:], (Iyj1/Tj1-Iyj0/Tj0)[None,None,:], (Ixj1/Tj1-Ixj0/Tj0)[None,None,:], (Ixyj1/Tj1-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj2[None,None,:]-xj0[None,None,:], yj2[None,None,:]-yj0[None,None,:], (Iyj2/Tj2-Iyj0/Tj0)[None,None,:], (Ixj2/Tj2-Ixj0/Tj0)[None,None,:], (Ixyj2/Tj2-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj3[None,None,:]-xj0[None,None,:], yj3[None,None,:]-yj0[None,None,:], (Iyj3/Tj3-Iyj0/Tj0)[None,None,:], (Ixj3/Tj3-Ixj0/Tj0)[None,None,:], (Ixyj3/Tj3-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj4[None,None,:]-xj0[None,None,:], yj4[None,None,:]-yj0[None,None,:], (Iyj4/Tj4-Iyj0/Tj0)[None,None,:], (Ixj4/Tj4-Ixj0/Tj0)[None,None,:], (Ixyj4/Tj4-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj5[None,None,:]-xj0[None,None,:], yj5[None,None,:]-yj0[None,None,:], (Iyj5/Tj5-Iyj0/Tj0)[None,None,:], (Ixj5/Tj5-Ixj0/Tj0)[None,None,:], (Ixyj5/Tj5-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj6[None,None,:]-xj0[None,None,:], yj6[None,None,:]-yj0[None,None,:], (Iyj6/Tj6-Iyj0/Tj0)[None,None,:], (Ixj6/Tj6-Ixj0/Tj0)[None,None,:], (Ixyj6/Tj6-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj7[None,None,:]-xj0[None,None,:], yj7[None,None,:]-yj0[None,None,:], (Iyj7/Tj7-Iyj0/Tj0)[None,None,:], (Ixj7/Tj7-Ixj0/Tj0)[None,None,:], (Ixyj7/Tj7-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj8[None,None,:]-xj0[None,None,:], yj8[None,None,:]-yj0[None,None,:], (Iyj8/Tj8-Iyj0/Tj0)[None,None,:], (Ixj8/Tj8-Ixj0/Tj0)[None,None,:], (Ixyj8/Tj8-Ixyj0/Tj0)[None,None,:])), np.hstack(( xj9[None,None,:]-xj0[None,None,:], yj9[None,None,:]-yj0[None,None,:], (Iyj9/Tj9-Iyj0/Tj0)[None,None,:], (Ixj9/Tj9-Ixj0/Tj0)[None,None,:], (Ixyj9/Tj9-Ixyj0/Tj0)[None,None,:])))).T.transpose((0,2,1))

    at=a.transpose((0,2,1))

    ata = np.matmul(at,a)

    ones = np.tile(np.identity(5),(len(ata),1,1))

    coefs = np.matmul(np.linalg.inv(ata+1e-12*ones),at)

    mesh["lsq_coefs"]=coefs
    
    return

def lin_coefs(mesh):

    mesh["lin_coefs"]={}

    ov = mesh["no_2nd_neigh"]
    ov1= mesh["no_1st_neigh"]

    zz = np.zeros((3,3))

    x0 = mesh["xj"]
    y0 = mesh["yj"]
    x1 = mesh["xj"][mesh["Neighboors"][0,:]]
    x2 = mesh["xj"][mesh["Neighboors"][1,:]]
    y1 = mesh["yj"][mesh["Neighboors"][0,:]]
    y2 = mesh["yj"][mesh["Neighboors"][1,:]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov1]=zz
    mesh["lin_coefs"]["12"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][1,:]]
    x2 = mesh["xj"][mesh["Neighboors"][2,:]]
    y1 = mesh["yj"][mesh["Neighboors"][1,:]]
    y2 = mesh["yj"][mesh["Neighboors"][2,:]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov1]=zz
    mesh["lin_coefs"]["23"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][2,:]]
    x2 = mesh["xj"][mesh["Neighboors"][0,:]]
    y1 = mesh["yj"][mesh["Neighboors"][2,:]]
    y2 = mesh["yj"][mesh["Neighboors"][0,:]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov1]=zz
    mesh["lin_coefs"]["31"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][0,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,0,0]]
    y1 = mesh["yj"][mesh["Neighboors"][0,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,0,0]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov]=zz
    mesh["lin_coefs"]["111"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][0,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,0,1]]
    y1 = mesh["yj"][mesh["Neighboors"][0,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,0,1]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov]=zz
    mesh["lin_coefs"]["112"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][1,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,1,0]]
    y1 = mesh["yj"][mesh["Neighboors"][1,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,1,0]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov]=zz
    mesh["lin_coefs"]["221"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][1,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,1,1]]
    y1 = mesh["yj"][mesh["Neighboors"][1,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,1,1]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov]=zz
    mesh["lin_coefs"]["222"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][2,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,2,0]]
    y1 = mesh["yj"][mesh["Neighboors"][2,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,2,0]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov]=zz
    mesh["lin_coefs"]["331"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][2,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,2,1]]
    y1 = mesh["yj"][mesh["Neighboors"][2,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,2,1]]

    coefs = np.array([[((y2-y0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((y0-y1)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()],[((x0-x2)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get(),((x1-x0)/((x1-x0)*(y2-y0)-(y1-y0)*(x2-x0))).get()]]).transpose((2,0,1))
    coefs[ov]=zz
    mesh["lin_coefs"]["332"]=coefs
    
    return

def lin_coefs2(mesh):

    mesh["lin_coefs2"]={}

    ov = mesh["no_2nd_neigh"]
    ov1= mesh["no_1st_neigh"]

    zz = np.array([[0,0,0],[0,0,0]])

    x0 = mesh["xj"]
    y0 = mesh["yj"]
    x1 = mesh["xj"][mesh["Neighboors"][0,:]]
    x2 = mesh["xj"][mesh["Neighboors"][1,:]]
    x3 = mesh["xj"][mesh["Neighboors"][2,:]]
    y1 = mesh["yj"][mesh["Neighboors"][0,:]]
    y2 = mesh["yj"][mesh["Neighboors"][1,:]]
    y3 = mesh["yj"][mesh["Neighboors"][2,:]]
    
    a = np.array([[(x1-x0).get(),(y1-y0).get()],[(x2-x0).get(),(y2-y0).get()],[(x3-x0).get(),(y3-y0).get()]]).transpose((2,0,1))

    at=a.transpose((0,2,1))

    ata = np.matmul(at,a)

    ones = np.tile(np.identity(2),(len(ata),1,1))

    coefs = np.matmul(np.linalg.inv(ata+1e-10*ones),at)

    coefs[ov1|ov]=zz

    mesh["lin_coefs2"]["T1"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][0,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,0,0]]
    x3 = mesh["xj"][mesh["Neighs2"][:,0,1]]
    y1 = mesh["yj"][mesh["Neighboors"][0,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,0,0]]
    y3 = mesh["yj"][mesh["Neighs2"][:,0,1]]
    
    a = np.array([[(x1-x0).get(),(y1-y0).get()],[(x2-x0).get(),(y2-y0).get()],[(x3-x0).get(),(y3-y0).get()]]).transpose((2,0,1))

    at=a.transpose((0,2,1))

    ata = np.matmul(at,a)

    ones = np.tile(np.identity(2),(len(ata),1,1))

    coefs = np.matmul(np.linalg.inv(ata+1e-10*ones),at)

    coefs[ov1|ov]=zz

    mesh["lin_coefs2"]["T2"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][1,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,1,0]]
    x3 = mesh["xj"][mesh["Neighs2"][:,1,1]]
    y1 = mesh["yj"][mesh["Neighboors"][1,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,1,0]]
    y3 = mesh["yj"][mesh["Neighs2"][:,1,1]]
    
    a = np.array([[(x1-x0).get(),(y1-y0).get()],[(x2-x0).get(),(y2-y0).get()],[(x3-x0).get(),(y3-y0).get()]]).transpose((2,0,1))

    at=a.transpose((0,2,1))

    ata = np.matmul(at,a)

    ones = np.tile(np.identity(2),(len(ata),1,1))

    coefs = np.matmul(np.linalg.inv(ata+1e-10*ones),at)

    coefs[ov1|ov]=zz

    mesh["lin_coefs2"]["T3"]=coefs

    x1 = mesh["xj"][mesh["Neighboors"][2,:]]
    x2 = mesh["xj"][mesh["Neighs2"][:,2,0]]
    x3 = mesh["xj"][mesh["Neighs2"][:,2,1]]
    y1 = mesh["yj"][mesh["Neighboors"][2,:]]
    y2 = mesh["yj"][mesh["Neighs2"][:,2,0]]
    y3 = mesh["yj"][mesh["Neighs2"][:,2,1]]
    
    a = np.array([[(x1-x0).get(),(y1-y0).get()],[(x2-x0).get(),(y2-y0).get()],[(x3-x0).get(),(y3-y0).get()]]).transpose((2,0,1))

    at=a.transpose((0,2,1))

    ata = np.matmul(at,a)

    ones = np.tile(np.identity(2),(len(ata),1,1))

    coefs = np.matmul(np.linalg.inv(ata+1e-10*ones),at)

    coefs[ov1|ov]=zz

    mesh["lin_coefs2"]["T4"]=coefs
    
    return

def precomp_weno(mesh):
    
    neighbor2(mesh)
    second_moments(mesh)
    in_point3(mesh)
    bathymetry_mid_point2(mesh)
    bathymetry_in_point3(mesh)

    lsq_coefs(mesh)
    lin_coefs(mesh)

    return

def precomp_weno2(mesh):
    
    neighbor2(mesh)
    in_point3(mesh)
    second_moments2(mesh)

    inner_outer_midpoint(mesh)

    lsq_coefs(mesh)
    lin_coefs2(mesh)

    bathymetry_mid_point2(mesh)
    bathymetry_in_point3(mesh)

    return



def bathymetry_mid_point(Bj, Bjk, xj, yj, xjk, yjk, jk, ijk, DRY,option):
    """
    """
    if option == "lbm":
        Hi0 = Bjk[0,:]
        Hi1 = Bjk[1,:]
        Hi2 = Bjk[2,:]
        
        Bx = ( Hi1*yjk[0,:] - Hi2*yjk[0,:] + Hi2*yjk[1,:] - Hi0*yjk[1,:] + Hi0*yjk[2,:] - Hi1*yjk[2,:])/(xjk[1,:]*yjk[0,:] - xjk[2,:]*yjk[0,:] - xjk[0,:]*yjk[1,:] + xjk[2,:]*yjk[1,:] + xjk[0,:]*yjk[2,:] - xjk[1,:]*yjk[2,:])
        By = (-Hi1*xjk[0,:] + Hi2*xjk[0,:] - Hi2*xjk[1,:] + Hi0*xjk[1,:] - Hi0*xjk[2,:] + Hi1*xjk[2,:])/(xjk[1,:]*yjk[0,:] - xjk[2,:]*yjk[0,:] - xjk[0,:]*yjk[1,:] + xjk[2,:]*yjk[1,:] + xjk[0,:]*yjk[2,:] - xjk[1,:]*yjk[2,:])

        Bml    = PieceWiseReconstruction.set_linear_midpoint_values(Bj, Bx, By, xj, yj, xjk, yjk)

    else:
        Bx = zeros_like(Bj)
        By = zeros_like(Bj)
        Bml = np.array([Bj, Bj, Bj])
        
    Bmr = Bml.take(jk)

    #Well-balanced reconstruction
    Bmj = np.maximum(Bml, Bmr)

    #Fix cells that have no neighbors
    Bmj[ijk < 0] = Bml[ijk < 0]

    return Bmj, Bx, By

def render_data(mesh, name):
    """
    This function renders the information provided in mesh

    Parameters
    ----------
        mesh : Dictionary that contains the information to be rendered
        name : String that specifies what variable to be rendered

    Returns
    -------
        File to be open using ParaView located at mesh["FilePath"]["Directory"]
    """
    filepath = mesh["FilePath"]["Directory"] + "/Paraview"
    if name.upper() in "BATHYMETRY":
        FileSaver.save_bathymetry(mesh)
    elif name.upper() in "WATERSURFACE":
        FileSaver.save_animation(mesh)
        mesh["s"] = 0
    else:
        info = debug_info(2) 
        print('\x1B[33m ALERT\x1B[0m[line=%d]: The provided option can not be rendered!'%info.lineno)
    return

def render_data_analytic(mesh, name):
    """
    This function renders the information provided in mesh

    Parameters
    ----------
        mesh : Dictionary that contains the information to be rendered
        name : String that specifies what variable to be rendered

    Returns
    -------
        File to be open using ParaView located at mesh["FilePath"]["Directory"]
    """
    filepath = mesh["FilePath"]["Directory"] + "/Paraview"
    if name.upper() in "BATHYMETRY":
        FileSaver.save_bathymetry(mesh)
    elif name.upper() in "WATERSURFACE":
        FileSaver.save_animation_analytic(mesh)
        mesh["s"] = 0
    else:
        info = debug_info(2) 
        print('\x1B[33m ALERT\x1B[0m[line=%d]: The provided option can not be rendered!'%info.lineno)
    return

def debug_info(level):
    """
    Provides information regarding the function called\n

    Parameters
    ----------
    level : int
        Level inside the dictionary to look for information

    Returns
    -------
    info : struc
        Structure containing information of the function called
    """
    callerframerecord = inspect.stack()[level]
    frame = callerframerecord[0]
    info  = inspect.getframeinfo(frame)

    return info

def compute_side_length(Elements, Coordinates, i, j):
    """
    Computes the length of the sides between the i-th and j-th node respectively

    Parameters
    ----------
        Elements : Conectivity array of Elements, [3, nElem]
        Coordinates : Node coordinates array of each node, [2, nNode]
        i : The first (i-th) node of the ij side, [1,1]
        j : The first (j-th) node of the ij side, [1,1]

    Returns
    -------
        lij : The length of the sides between nodes i,j
    """
    u = Coordinates[:,Elements[j,:]] - Coordinates[:,Elements[i,:]]
    lij = np.sqrt(u[0,:]**2 + u[1,:]**2)

    return lij

def compute_tangent_vector(Elements, Coordinates, i, j):
    """
    Returns the element normalized tangent vector for each element

    Parameters
    ----------
        Elements : Conectivity array of Elements, [3, nElem]
        Coordinates : Node coordinates array of each node, [2, nNode]
        i : The first (i-th) node of the ij side, [1,1]
        j : The first (j-th) node of the ij side, [1,1]

    Returns
    -------
        t : The tangent vector of the ij-edge, [2, nElem]
        
    """
    u = Coordinates[:,Elements[j,:]] - Coordinates[:,Elements[i,:]]
    L = np.sqrt(u[0,:]**2 + u[1,:]**2)
    t = np.divide(u, L) 

    return t

def compute_element_normals(Elements, Coordinates):
    """
    Returns the element normal components for each element

    Parameters
    ----------
        Elements : Conectivity array of Elements, [3, nElem]
        Coordinates : Node coordinates array of each node, [2, nNode]

    Returns
    -------
        nx : Unit-Normal component over X, [3, nElem]
        ny : Unit-Normal component over Y, [3, nElem]
    """
    #Tangent vectors definition:
    u1 = compute_tangent_vector(Elements, Coordinates, 0, 1)
    u2 = compute_tangent_vector(Elements, Coordinates, 1, 2)
    u3 = compute_tangent_vector(Elements, Coordinates, 2, 0)

    #Normal vector definition
    nx = np.array([ u1[1,:], u2[1,:], u3[1,:]])
    ny = np.array([-u1[0,:],-u2[0,:],-u3[0,:]])

    return nx, ny 

def compute_element_edge_lengths(Elements, Coordinates):
    """
    Computes the element edge sides length in order

    Parameters
    ----------
        Elements : Conectivity array of Elements, [3, nElem]
        Coordinates : Node coordinates array of each node, [2, nNode]

    Returns
    -------
        ljk : Side lengths for each triangle, [3, nElem]
    """
    #Tangent vector sides (12, 23, and 31)
    L1 = compute_side_length(Elements, Coordinates, 0, 1)
    L2 = compute_side_length(Elements, Coordinates, 1, 2)
    L3 = compute_side_length(Elements, Coordinates, 2, 0)

    #Side's length
    ljk = np.array([L1,L2,L3])

    return ljk

def compute_element_areas(Elements, Coordinates):
    """
    Computes the area for each triangle using Heron's formula

    Parameters
    ----------
        Elements : Conectivity array of Elements, [3, nElem]
        Coordinates : Node coordinates array of each node, [2, nNode]

    Returns
    -------
        Tj : Area of the j-th Triangle, [1, nElem]
    """
    #Triangle side lengths
    ljk = compute_element_edge_lengths(Elements, Coordinates)

    #Semi-perimeter of each triangle
    s = 0.5*ljk.sum(axis=0)

    #Heron's formula for area
    A = np.sqrt(s*(s - ljk[0,:])*(s - ljk[1,:])*(s - ljk[2,:]))

    return A

def compute_element_edge_altitudes(Elements, Coordinates):
    """
    Returns the altitude (in the geometric sense) for each triangular element

    Parameters
    ----------
        Elements : Conectivity array of Elements, [3, nElem]
        Coordinates : Node coordinates array of each node, [2, nNode]

    Returns
    -------
        rjk : Altitudes at mid-pints for each element, [3, nElem]
    """
    #Triangle's area
    Tj = compute_element_areas(Elements, Coordinates)

    #Triangle side lengths
    ljk = compute_element_edge_lengths(Elements, Coordinates)

    #Altitudes to each side:
    rjk = np.divide(2*Tj, ljk)
    
    return rjk
