import cupy as np
import scipy as sp
import cuSolver as Solver
import cuUtilities as Utilities
import cuFileSaver as FileSaver
import cuFileLoader as FileLoader
import cuBoundaryConditions as BoundaryConditions


def calculate_error(mesh, anal, params, t=0):

    if mesh["s"]*mesh["Constants"]["dt_save"] <= t:

        Wanal = anal(mesh,t,params)
        Wsim  = mesh["Wj"]
        dL    = mesh["Constants"]["dL"]
        A = mesh["Tj"]

        qError = A*(Wanal-Wsim)**2
        L2Error = np.sqrt(np.sum(qError))
        mesh["L2Error"].append(L2Error)
        mesh["s"]+=1

    return

#############################################################################################################
##          S O L I T A R Y   W A V E   P R O P A G A T I O N        ##    Ref.: Advection eq. solution    ##
#############################################################################################################

sech = lambda x: 2.0*np.exp(-1.0*x)/(1.0 + np.exp(-2.0*x))


def solitary_wave_jk(mesh, t, params):

    hmax = params[0]
    d    = params[1]
    x0   = params[2]

    g=mesh["Constants"]["Gravity"]
    gamma = np.sqrt( 0.75 * hmax/d**3) # [1/m]
    c = 39 #np.sqrt(g*(hmax + d))         
    
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]
    yjk         = Coordinates[1,Elements]

    Wjk = hmax * sech(gamma*(xjk - x0 - c*t))**2 + d

    return Wjk

def solitary_wave_j(mesh, t, params):

    hmax = params[0]
    d    = params[1]
    x0   = params[2]

    g=mesh["Constants"]["Gravity"]
    gamma = np.sqrt( 0.75 * hmax/d**3) # [1/m]
    c = 39 #np.sqrt(g*(hmax + d))         
    
    xjk         = mesh["xj"]
    yjk         = mesh["yj"]

    Wjk = hmax * sech(gamma*(xjk - x0 - c*t))**2 + d

    return Wjk


def run_solitary_wave(mesh, hmax, d, x0,p=1):
    
    dt=mesh["Constants"]["dt_save"]

    BoundaryConditions.impose(mesh)

    FileSaver.save_bathymetry(mesh)
    FileSaver.save_animation_analytic(mesh, 0)

    t  = 0.0
    i  = 0

    while (t < mesh["Constants"]["Tmax"]*p):
        t += dt
        #Solves for this time step
        mesh["Wjk"] = solitary_wave_jk(mesh, t, [hmax, d, x0])
        mesh["Wj"]  = solitary_wave_j(mesh, t, [hmax, d, x0])
        Solver.cutoff_values(mesh)
        BoundaryConditions.impose(mesh)

        #Save to disk the solution so far
        i += 1
        FileSaver.save_animation_analytic(mesh, t)

        print((" | Running: "+mesh["FilePath"]["Directory"]+" (Analytic) | Progress : %1.2f%s ( t = %1.5f, dt = %1.5f, it = %1d )" % (100*t/mesh["Constants"]["Tmax"]*p, "%", t, dt, i)), flush=True, end="\r")

    return



###############################################################################################################################
##                 W E T - W E T   D A M   B R E A K                 ##       Ref.: https://doi.org/10.1002/fld.3741         ##
###############################################################################################################################



def wet_dam_break_jk(mesh, t, params):

    H_l = params[0]
    H_r = params[1]
    cm  = params[2]
    x0  = params[3]

    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]
    yjk         = Coordinates[1,Elements]
    Bjk         = mesh["Bjk"]
    g           = mesh["Constants"]["Gravity"]

    cl = np.sqrt(g*H_l)
    cr = np.sqrt(g*H_r)

    xA = x0-cl*t
    xB = x0+t*(2*cl-3*cm)
    xC = x0+t*(2*(cm**2)*(cl-cm))/(cm**2-cr**2)

    #H = np.where(xjk<=xA,H_l,np.where((xA<=xjk)&(xjk<=xB),((xjk/t-2*cl)**2)/(9*g),np.where((xB<=xjk)&(xjk<=xC),cm**2/g,H_l)))

    #H = np.zeros_like(xjk)

    #H[xjk<=xA]=H_l
    #H[np.where((xjk>=xA)&(xjk<=xB))]=((xjk[(xjk>=xA)&(xjk<=xB)]/t-2*cl)**2)/(9*g)
    #H[np.where((xjk>=xB)&(xjk<=xC))]=cm**2/g
    #H[(xjk>xC)]=H_l

    #innermost=np.where((xB<=xjk)&(xjk<=xC),cm**2/g,H_r)
    #inner=np.where((xA<xjk)&(xjk<xB),((xjk/t-2*cl)**2)/(9*g),innermost)
    #H=np.where(xjk<=xA,H_l,inner)

    H=np.piecewise(xjk,[xjk<=xA,(xjk>=xA)&(xjk<=xB),(xjk>xB)&(xjk<xC),xjk>=xC],[H_l,lambda x: ((x/t-2*cl)**2)/(9*g),cm**2/g,H_r])

    return Bjk + H

def wet_dam_break_j(mesh, t, params):

    H_l = params[0]
    H_r = params[1]
    x0  = params[2]
    cm  = params[3]

    xjk         = mesh["xj"]
    yjk         = mesh["yj"]
    Bjk         = mesh["Bj"]
    g           = mesh["Constants"]["Gravity"]

    cl = np.sqrt(g*H_l)
    cr = np.sqrt(g*H_r)

    xA = x0-cl*t
    xB = x0+t*(2*cl-3*cm)
    xC = x0+t*(2*(cm**2)*(cl-cm))/(cm**2-cr**2)

    #H = np.where(xjk<=xA,H_l,np.where((xA<=xjk)&(xjk<=xB),((xjk/t-2*cl)**2)/(9*g),np.where((xB<=xjk)&(xjk<=xC),cm**2/g,H_l)))

    H = np.zeros_like(xjk)

    H[xjk<=xA]=H_l
    H[np.where((xjk>=xA)&(xjk<=xB))]=((xjk[(xjk>=xA)&(xjk<=xB)]/t-2*cl)**2)/(9*g)
    H[np.where((xjk>=xB)&(xjk<=xC))]=cm**2/g
    H[(xjk>xC)]=H_r

    #innermost=np.where((xB<=xjk)&(xjk<=xC),cm**2/g,H_r)
    #inner=np.where((xA<xjk)&(xjk<xB),((xjk/t-2*cl)**2)/(9*g),innermost)
    #H=np.where(xjk<=xA,H_l,inner)
    
    #H=np.piecewise(xjk,[xjk<=xA,(xjk>=xA)&(xjk<=xB),(xjk>xB)&(xjk<xC),xjk>=xC],[H_l,lambda x: ((x/t-2*cl)**2)/(9*g),cm**2/g,H_r])

    return Bjk + H

def old_wet_dam_break(xjk, yjk, Bjk, g, t, H_l, H_r, cm, x0):

    #g=np.float64(g)
    #t=np.float64(t)
    #two=np.float64(2)
    #nine=np.float64(9)

    #xjk=np.float64(xjk)
    #yjk=np.float64(yjk)

    cl = np.sqrt(g*H_l)
    cr = np.sqrt(g*H_r)

    xA = x0-cl*t
    xB = x0+t*(2.*cl-3.*cm)
    xC = x0+t*(2.*(cm**2)*(cl-cm))/(cm**2-cr**2)

    innermost=np.where((xB<=xjk)&(xjk<=xC),cm**2/g,H_r)
    inner=np.where((xA<=xjk)&(xjk<=xB),((xjk/t-2*cl)**2)/(9*g),innermost)
    H=np.where(xjk<=xA,H_l,inner)

    #H = np.where(xjk<=xA,H_l,np.where((xA<=xjk)&(xjk<=xB),((xjk/t-2*cl)**2)/(9*g),np.where((xB<=xjk)&(xjk<=xC),cm**2/g,H_l)))

    #if xjk<=xA:
    #    H = H_l
    #elif xjk<=xB and xjk>xA:
    #    H = np.divide(((np.subtract(np.divide(xjk,t),two*cl))*(np.subtract(np.divide(xjk,t),two*cl))),(nine*g))
    #elif xB<xjk and xjk<=xC:
    #    H = cm**2/g
    #else:
    #    H = H_r

    return Bjk + H
    
wet_dam_breakv=np.vectorize(old_wet_dam_break)

def run_wet_dam_break(mesh, H_l, H_r, x0=0,tstart=0,tend=0,p=1):

    g           = mesh["Constants"]["Gravity"]
    Coordinates = mesh["Coordinates"]
    Elements    = mesh["Elements"]
    xjk         = Coordinates[0,Elements]
    yjk         = Coordinates[1,Elements]
    Bjk         = mesh["Bjk"]
    xj          = mesh["xj"]
    yj          = mesh["yj"]
    Bj          = mesh["Bj"]
    dt          = mesh["Constants"]["dt_save"]

    BoundaryConditions.impose(mesh)

    FileSaver.save_bathymetry(mesh)
    FileSaver.save_animation_analytic(mesh, 0)

    t  = tstart-dt
    i  = 0

    if tend:
        mesh["Constants"]["Tmax"]=tend

    mesh["s"] = int((t+dt)//mesh["Constants"]["dt_save"])

    cl = np.sqrt(g*H_l)
    cr = np.sqrt(g*H_r)
    
    pol = lambda x: x**6-8*cl**2*x**2*cr**2+16*cl*x**3*cr**2-9*x**4*cr**2-x**2*cr**4+cr**6

    cm = sp.optimize.bisect(pol,cl,cr)

    while (t < mesh["Constants"]["Tmax"]*p):
        t += dt
        #Solves for this time step
        mesh["Wjk"] = wet_dam_break_jk(mesh, t, [H_l, H_r, x0, cm])
        mesh["Wj"]  = wet_dam_break_j(mesh, t, [H_l, H_r, x0, cm])
        Solver.cutoff_values(mesh)
        BoundaryConditions.impose(mesh)

        #Save to disk the solution so far
        i += 1
        FileSaver.save_animation_analytic(mesh, t)

        print((" | GPU | Running: "+mesh["FilePath"]["Directory"]+" (Analytic) | Progress : %1.2f%s ( t = %1.5f, dt = %1.5f, it = %1d )" % (100*t/mesh["Constants"]["Tmax"]*p, "%", t, dt, i)), flush=True, end="\r")

    return