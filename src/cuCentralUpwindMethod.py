import cupy as np

def set_water_depth(Wmj, Bmj, DRY):
    """
    This function computes the water deph at the midpoints of the triangle
    
    Parameters
    ----------
        Wmj : Mid-points values of water surface, [3, nElem]
        Bmj : Mid-point values of bathymetry, [3, nElem]
        DRY : Tolerance that defines when a cell is dry, [1,1]
    
    Returns
    -------
        Hmj : Mid-point water depth values of quatity Hj, [3, nElem]
    """
    #Water depth variable at mid-points
    Hmj = np.maximum(0.01*DRY, Wmj - Bmj)

    return Hmj

def midvelocity(XUmj, Xmj, DRY):
    """
    This function desingularizes the velocities for an specified tolerance

    Parameters
    ----------
        Xmj : Mid-point water depth, [3, nElem]
        TOL : Tolerance that defines when a number is zeroed-out, [1,1]
    
    Returns
    -------
        XUmj : Corrected Mid-side value of Flux, [3, nElem]
        umj  : Inside mid-point values of the velocity, [3, nElem]
    """
    #Mid-Point Edge 12 
    AUX = Xmj[0,:]**4
    uj1 = np.divide(np.sqrt(2)*Xmj[0,:]*XUmj[0,:], np.sqrt(AUX + np.maximum(AUX, DRY)))

    #Mid-Point Edge 23 
    AUX = Xmj[1,:]**4
    uj2 = np.divide(np.sqrt(2)*Xmj[1,:]*XUmj[1,:], np.sqrt(AUX + np.maximum(AUX, DRY)))

    #Mid-Point Edge 31
    AUX = Xmj[2,:]**4
    uj3 = np.divide(np.sqrt(2)*Xmj[2,:]*XUmj[2,:], np.sqrt(AUX + np.maximum(AUX, DRY)))

    #Flux Correction
    umj = np.array([uj1, uj2, uj3])
    XUmj = Xmj*umj
     
    return XUmj, umj

def one_sided_speed(Hmj, umj, vmj, nx, ny, jk, g):
    """
    This function compute the inward and outward one-sided velocities

    Parameters
    ----------
        Hmj : Mid-side value of water depth, [3, nElem]
        ujm : Normal Velocity over X-direction at mid-pints, [3, nElem]
        vjm : Normal Velocity over Y-direction at mid-pints, [3, nElem]
        nx  : Unit-Normal component over X, [3, nElem]
        ny  : Unit-Normal component over Y, [3, nElem]
        jk  : Global indeces for side neighbor cells quantities, [1, 3*nElem]
        g   : Gravity constant, [1,1]
    
    Returns
    -------
        a_in  : One-sided inwards velocity, [3, nElem]
        a_out : One-sided outwards velocity, [3, nElem] 
    """
    #Adjacent state variable values:
    Hmjk = Hmj.take(jk)
    umjk = umj.take(jk)
    vmjk = vmj.take(jk)

    #One-Sided Speed of propagation
    uj   =  nx*umj  + ny*vmj    
    ujk  =  nx*umjk + ny*vmjk 

    #Outward Speed Propagation
    a_out = np.maximum(uj + np.sqrt(g*Hmj), ujk + np.sqrt(g*Hmjk))
    a_out = np.maximum(a_out, 0.0)

    #Inward Speed Propagation
    a_in  =  np.minimum(uj - np.sqrt(g*Hmj), ujk - np.sqrt(g*Hmjk))
    a_in  = -np.minimum(a_in, 0.0)

    return a_out, a_in

def one_sided_speed2(Hmj1, umj1, vmj1, Hmj2, umj2, vmj2, Hmj1out, umj1out, vmj1out, Hmj2out, umj2out, vmj2out, nx, ny, jk, g):
    """
    This function compute the inward and outward one-sided velocities

    Parameters
    ----------
        Hmji   : Water depth at mid-point i, [3, nElem]
        ujmi   : Normal Velocity over X-direction at mid-point i, [3, nElem]
        vjmi   : Normal Velocity over Y-direction at mid-point i, [3, nElem]
        nx     : Unit-Normal component over X, [3, nElem]
        ny     : Unit-Normal component over Y, [3, nElem]
        jk     : Global indeces for side neighbor cells quantities, [1, 3*nElem]
        choice : 0 at position [i,j] if Xmj1[i,j] is located at the same coordinates as Xmj1.take(jk)[i,j] and 1 otherwise, [3, nElem]
        g      : Gravity constant, [1,1]
    
    Returns
    -------
        a_in  : One-sided inwards velocity, [3, nElem]
        a_out : One-sided outwards velocity, [3, nElem] 
    """
    #Adjacent state variable values:
    Hmjk1 = Hmj1out
    umjk1 = umj1out
    vmjk1 = vmj1out

    Hmjk2 = Hmj2out
    umjk2 = umj2out
    vmjk2 = vmj2out

    #One-Sided Speed of propagation
    uj1   =  nx*umj1  + ny*vmj1    
    ujk1  =  nx*umjk1 + ny*vmjk1
    uj2   =  nx*umj2  + ny*vmj2    
    ujk2  =  nx*umjk2 + ny*vmjk2 

    #Outward Speed Propagation
    a_out1 = np.maximum(uj1 + np.sqrt(g*Hmj1), ujk1 + np.sqrt(g*Hmjk1))
    #a_out1 = np.maximum(a_out1, 0.0)
    a_out2 = np.maximum(uj2 + np.sqrt(g*Hmj2), ujk2 + np.sqrt(g*Hmjk2))
    #a_out2 = np.maximum(a_out2, 0.0)

    a_out = np.maximum(a_out1,a_out2)
    a_out = np.maximum(a_out,0.0)

    #Inward Speed Propagation
    a_in1  =  np.minimum(uj1 - np.sqrt(g*Hmj1), ujk1 - np.sqrt(g*Hmjk1))
    #a_in1  = -np.minimum(a_in1, 0.0)
    a_in2  =  np.minimum(uj2 - np.sqrt(g*Hmj2), ujk2 - np.sqrt(g*Hmjk2))
    #a_in2  = -np.minimum(a_in2, 0.0)

    a_in = np.minimum(a_in1,a_in2)
    a_in = -np.minimum(a_in,0.0)

    #a_out=(a_out1+a_out2)/2
    #a_in =(a_in1+a_in2)/2

    return a_out, a_in

def source_term(Hmj, wx, wy, ljk, Tj, nx, ny, g):
    """
    This function computes the Source term components

    Parameters
    ----------
        Hmj : Water depth at mid-point sides, [3, nElem]
        ljk : Side lengths for each triangle, [3, nElem]
        Tj  : Area of the j-th Triangle, [1, nElem]
        nx  : Unit-Normal component over X, [3, nElem]
        ny  : Unit-Normal component over Y, [3, nElem]
        g   : Gravity constant, [1,1]
    
    Returns
    -------
        Sx : Source term along-X, [1, nElem]
        Sy : Source term along-Y, [1, nElem]
    """
    #Auxiliar variables
    aux12 = ljk[0,:]*Hmj[0,:]**2
    aux23 = ljk[1,:]*Hmj[1,:]**2
    aux31 = ljk[2,:]*Hmj[2,:]**2

    #Source Term Values
    Sx = g*(aux12*nx[0,:] + aux23*nx[1,:] + aux31*nx[2,:])/(2.0*Tj) - g*wx*(Hmj[0,:]+Hmj[1,:]+Hmj[2,:])/3
    Sy = g*(aux12*ny[0,:] + aux23*ny[1,:] + aux31*ny[2,:])/(2.0*Tj) - g*wy*(Hmj[0,:]+Hmj[1,:]+Hmj[2,:])/3

    return Sx, Sy

def source_term2(Hmj1,Hmj2, Wj1, Wj2, Wj3, Bj1, Bj2, Bj3, wx1, wx2, wx3, wy1, wy2, wy3, ljk, Tj, nx, ny, g):
    """
    This function computes the Source term components

    Parameters
    ----------
        Hmj : Water depth at mid-point sides, [3, nElem]
        ljk : Side lengths for each triangle, [3, nElem]
        Tj  : Area of the j-th Triangle, [1, nElem]
        nx  : Unit-Normal component over X, [3, nElem]
        ny  : Unit-Normal component over Y, [3, nElem]
        g   : Gravity constant, [1,1]
    
    Returns
    -------
        Sx : Source term along-X, [1, nElem]
        Sy : Source term along-Y, [1, nElem]
    """
    #Auxiliar variables
    aux121 = 0.5*ljk[0,:]*Hmj1[0,:]**2
    aux231 = 0.5*ljk[1,:]*Hmj1[1,:]**2
    aux311 = 0.5*ljk[2,:]*Hmj1[2,:]**2
    aux122 = 0.5*ljk[0,:]*Hmj2[0,:]**2
    aux232 = 0.5*ljk[1,:]*Hmj2[1,:]**2
    aux312 = 0.5*ljk[2,:]*Hmj2[2,:]**2

    #Source Term Values
    Sx = g*((aux121+aux122)*nx[0,:] + (aux231+aux232)*nx[1,:] + (aux311+aux312)*nx[2,:])/(2.0*Tj) - g*(wx1*(Wj1-Bj1)+wx2*(Wj2-Bj2)+wx3*(Wj3-Bj3))/3 #Debería reconstruir mejor
    Sy = g*((aux121+aux122)*ny[0,:] + (aux231+aux232)*ny[1,:] + (aux311+aux312)*ny[2,:])/(2.0*Tj) - g*(wy1*(Wj1-Bj1)+wy2*(Wj2-Bj2)+wy3*(Wj3-Bj3))/3

    return Sx, Sy

def source_term_bestia(h,bx,by,g):

    #Source Term Values
    Sx = -g*h*bx
    Sy = -g*h*by

    return Sx, Sy

def coriolis(hu,hv,coriolis):

    f=coriolis

    fx = f*hv
    fy = -f*hu

    return fx, fy

def friction_term(HUmj, HVmj, Wj, Bj, n, g, DRY):
    """
    This function computes the Friction term components

    Parameters
    ----------
        Bj  : Bathymetry values located at the mass-center, [1, nElem]
        Wj  : Water level values located at the mass-center, [1, nElem]
        ujm : Normal Velocity over X-direction at mid-pints, [3, nElem]
        vjm : Normal Velocity over Y-direction at mid-pints, [3, nElem]
        g   : Gravity constant, [1,1]
  	    n   : Manning's roughness coefficient, [1,nElem]
    
    Returns
    -------
        Fr : Friction term along-X and -Y direction, [1, nElem]
    """
    #Water depth values at Cell center 
    Hj = Wj - Bj
    Hj[Hj<DRY] = DRY

    #Velocity components
    hu = (HUmj[0,:] + HUmj[1,:] + HUmj[2,:])/3.0
    hv = (HVmj[0,:] + HVmj[1,:] + HVmj[2,:])/3.0

    #Mannin'g friction variable
    Fr = g*n**2*np.sqrt(hu**2 + hv**2)/(Hj**(7.0/3.0))

    return Fr

def water_level_function(Wmj, HUmj, HVmj, a_in, a_out, ljk, nx, ny, jk):
    """
    Computes the inside and outside conserved variables function for the first vector component

    Parameters
    ----------
        Wmj   : Water depth at mid-point sides, [3, nElem]
        HUmj  : Flux along-X at mid-point sides, [3, nElem]
        HVmj  : Flux along-Y at mid-point sides, [3, nElem]
        a_in  : Inwards one-sides propagation velocity, [3, nElem]
        a_out : Outwards one-sides propagation velocity, [3, nElem]
        nx    : Unit-Normal component over X, [3, nElem]
        ny    : Unit-Normal component over Y, [3, nElem]
        ljk   : Side lengths for each triangle, [3, nElem]
        jk    : Global indeces for side neighbor cells quantities, [1, 3*nElem]
    
    Returns
    -------
        Uin   : Inward mid-point values of the Flux Function, [1, nElem]
        Uout  : Outwards mid-point values of the Flux Function, [1, nElem]
        FUin  : Inside mid-point values of the Flux Function, [1, nElem]
        FUout : Outside mid-point values of the Flux Function, [1, nElem]
        GUin  : Inside mid-point values of the Flux Function, [1, nElem]
        GUout : Outside mid-point values of the Flux Function, [1, nElem]
    """
    #Adjacent Water Level state variable values
    Hin  = Wmj
    Hout = Hin.take(jk)
    Coeff = ljk*a_in*a_out/(a_in + a_out)

    Uin  = Coeff[0,:]*Hin[0,:]  + Coeff[1,:]*Hin[1,:]  + Coeff[2,:]*Hin[2,:]
    Uout = Coeff[0,:]*Hout[0,:] + Coeff[1,:]*Hout[1,:] + Coeff[2,:]*Hout[2,:]

    #Adjacent Flux-Discharge along-X state variable values
    Hin  = HUmj
    Hout = Hin.take(jk)
    Coeff = ljk*nx/(a_in + a_out)

    FUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    FUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    #Adjacent Flux-Discharge along-Y state variable values
    Hin  = HVmj
    Hout = Hin.take(jk)
    Coeff = ljk*ny/(a_in + a_out)

    GUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    GUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    return Uin, Uout, FUin, FUout, GUin, GUout

def water_level_function2(Wmj, HUmj, HVmj, Wmjout, HUmjout, HVmjout, a_in, a_out, ljk, nx, ny, jk):
    """
    Computes the inside and outside conserved variables function for the first vector component

    Parameters
    ----------
        Wmji   : Water depth at mid-point side i, [3, nElem]
        HUmji  : Flux along-X at mid-point side i, [3, nElem]
        HVmji  : Flux along-Y at mid-point side i, [3, nElem]
        a_in   : Inwards one-sides propagation velocity, [3, nElem]
        a_out  : Outwards one-sides propagation velocity, [3, nElem]
        nx     : Unit-Normal component over X, [3, nElem]
        ny     : Unit-Normal component over Y, [3, nElem]
        ljk    : Side lengths for each triangle, [3, nElem]
        jk     : Global indeces for side neighbor cells quantities, [1, 3*nElem]
        choice : 0 at position [i,j] if Xmj1[i,j] is located at the same coordinates as Xmj1.take(jk)[i,j] and 1 otherwise, [3, nElem]
    
    Returns
    -------
        Uin   : Inward mid-point values of the Flux Function, [1, nElem]
        Uout  : Outwards mid-point values of the Flux Function, [1, nElem]
        FUin  : Inside mid-point values of the Flux Function, [1, nElem]
        FUout : Outside mid-point values of the Flux Function, [1, nElem]
        GUin  : Inside mid-point values of the Flux Function, [1, nElem]
        GUout : Outside mid-point values of the Flux Function, [1, nElem]
    """

    #Adjacent Water Level state variable values
    Hin  = Wmj
    Hout = Wmjout
    Coeff = ljk*a_in*a_out/(a_in + a_out)

    Uin  = Coeff[0,:]*Hin[0,:]  + Coeff[1,:]*Hin[1,:]  + Coeff[2,:]*Hin[2,:]
    Uout = Coeff[0,:]*Hout[0,:] + Coeff[1,:]*Hout[1,:] + Coeff[2,:]*Hout[2,:]

    #Adjacent Flux-Discharge along-X state variable values
    Hin  = HUmj
    Hout = HUmjout
    Coeff = ljk*nx/(a_in + a_out)

    FUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    FUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    #Adjacent Flux-Discharge along-Y state variable values
    Hin  = HVmj
    Hout = HVmjout
    Coeff = ljk*ny/(a_in + a_out)

    GUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    GUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    return Uin, Uout, FUin, FUout, GUin, GUout

def flux_function_x(Hmj, HUmj, HVmj, a_in, a_out, umj, ljk, nx, ny, jk, g):
    """
    Computes the inside and outside conserved variables function for the second vector component

    Parameters
    ----------
        Hmj   : Water depth at mid-point sides, [3, nElem]
        HUmj  : Flux along-X at mid-point sides, [3, nElem]
        HVmj  : Flux along-Y at mid-point sides, [3, nElem]
        a_in  : Inwards one-sides propagation velocity, [3, nElem]
        a_out : Outwards one-sides propagation velocity, [3, nElem]
        umj   : Velocity along the X-direction, [3, nElem]
        nx    : Unit-Normal component over X, [3, nElem]
        ny    : Unit-Normal component over Y, [3, nElem]
        ljk   : Side lengths for each triangle, [3, nElem]
        jk    : Global indeces for side neighbor cells quantities, [1, 3*nElem]
    
    Returns
    -------
        Uin   : Inward mid-point values of the Flux Function, [1, nElem]
        Uout  : Outwards mid-point values of the Flux Function, [1, nElem]
        FUin  : Inside mid-point values of the Flux Function, [1, nElem]
        FUout : Outside mid-point values of the Flux Function, [1, nElem]
        GUin  : Inside mid-point values of the Flux Function, [1, nElem]
        GUout : Outside mid-point values of the Flux Function, [1, nElem]
    """
    #Adjacent state variable values:
    Hin  = HUmj
    Hout = Hin.take(jk)
    Coeff = ljk*a_in*a_out/(a_in + a_out)

    Uin  = Coeff[0,:]*Hin[0,:]  + Coeff[1,:]*Hin[1,:]  + Coeff[2,:]*Hin[2,:]
    Uout = Coeff[0,:]*Hout[0,:] + Coeff[1,:]*Hout[1,:] + Coeff[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HUmj*umj + 0.5*g*Hmj**2
    Hout = Hin.take(jk)
    Coeff = ljk*nx/(a_in + a_out)

    FUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    FUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HVmj*umj
    Hout = Hin.take(jk)
    Coeff = ljk*ny/(a_in + a_out)

    GUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    GUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    return Uin, Uout, FUin, FUout, GUin, GUout

def flux_function_x2(Hmj, HUmj, HVmj, Hmjout, HUmjout, HVmjout, a_in, a_out, umj, umjout, ljk, nx, ny, jk, g):
    """
    Computes the inside and outside conserved variables function for the second vector component

    Parameters
    ----------
        Hmji   : Water depth at mid-point side i, [3, nElem]
        HUmji  : Flux along-X at mid-point side i, [3, nElem]
        HVmji  : Flux along-Y at mid-point side i, [3, nElem]
        a_in   : Inwards one-sides propagation velocity, [3, nElem]
        a_out  : Outwards one-sides propagation velocity, [3, nElem]
        umji   : Velocity along the X-direction at mid-point side i, [3, nElem]
        nx     : Unit-Normal component over X, [3, nElem]
        ny     : Unit-Normal component over Y, [3, nElem]
        ljk    : Side lengths for each triangle, [3, nElem]
        jk     : Global indeces for side neighbor cells quantities, [1, 3*nElem]
        g      : Gravity constant
        choice : 0 at position [i,j] if Xmj1[i,j] is located at the same coordinates as Xmj1.take(jk)[i,j] and 1 otherwise, [3, nElem]
    
    Returns
    -------
        Uin   : Inward mid-point values of the Flux Function, [1, nElem]
        Uout  : Outwards mid-point values of the Flux Function, [1, nElem]
        FUin  : Inside mid-point values of the Flux Function, [1, nElem]
        FUout : Outside mid-point values of the Flux Function, [1, nElem]
        GUin  : Inside mid-point values of the Flux Function, [1, nElem]
        GUout : Outside mid-point values of the Flux Function, [1, nElem]
    """
    #Adjacent state variable values:
    Hin  = HUmj
    Hout = HUmjout
    Coeff = ljk*a_in*a_out/(a_in + a_out)

    Uin  = Coeff[0,:]*Hin[0,:]  + Coeff[1,:]*Hin[1,:]  + Coeff[2,:]*Hin[2,:]
    Uout = Coeff[0,:]*Hout[0,:] + Coeff[1,:]*Hout[1,:] + Coeff[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HUmj*umj + 0.5*g*Hmj**2
    Hout = HUmjout*umjout+0.5*g*Hmjout**2
    Coeff = ljk*nx/(a_in + a_out)

    FUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    FUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HVmj*umj
    Hout = HVmjout*umjout
    Coeff = ljk*ny/(a_in + a_out)

    GUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    GUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    return Uin, Uout, FUin, FUout, GUin, GUout

def flux_function_y(Hmj, HUmj, HVmj, a_in, a_out, vmj, ljk, nx, ny, jk, g):
    """
    Computes the inside and outside conserved variables function for the third vector component

    Parameters
    ----------
        Hmj   : Water depth at mid-point sides, [3, nElem]
        HUmj  : Flux along-X at mid-point sides, [3, nElem]
        HVmj  : Flux along-Y at mid-point sides, [3, nElem]
        a_in  : Inwards one-sides propagation velocity, [3, nElem]
        a_out : Outwards one-sides propagation velocity, [3, nElem]
        vmj   : Velocity along the Y-direction, [3, nElem]
        nx    : Unit-Normal component over X, [3, nElem]
        ny    : Unit-Normal component over Y, [3, nElem]
        ljk   : Side lengths for each triangle, [3, nElem]
        jk    : Global indeces for side neighbor cells quantities, [1, 3*nElem]
    
    Returns
    -------
        Uin   : Inward mid-point values of the Flux Function, [1, nElem]
        Uout  : Outwards mid-point values of the Flux Function, [1, nElem]
        FUin  : Inside mid-point values of the Flux Function, [1, nElem]
        FUout : Outside mid-point values of the Flux Function, [1, nElem]
        GUin  : Inside mid-point values of the Flux Function, [1, nElem]
        GUout : Outside mid-point values of the Flux Function, [1, nElem]
    """
    #Adjacent state variable values:
    Hin  = HVmj
    Hout = Hin.take(jk)
    Coeff = ljk*a_in*a_out/(a_in + a_out)

    Uin  = Coeff[0,:]*Hin[0,:]  + Coeff[1,:]*Hin[1,:]  + Coeff[2,:]*Hin[2,:]
    Uout = Coeff[0,:]*Hout[0,:] + Coeff[1,:]*Hout[1,:] + Coeff[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HUmj*vmj
    Hout = Hin.take(jk)
    Coeff = ljk*nx/(a_in + a_out)

    FUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    FUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HVmj*vmj + 0.5*g*Hmj**2
    Hout = Hin.take(jk)
    Coeff = ljk*ny/(a_in + a_out)

    GUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    GUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    return Uin, Uout, FUin, FUout, GUin, GUout

def flux_function_y2(Hmj, HUmj, HVmj, Hmjout, HUmjout, HVmjout, a_in, a_out, vmj, vmjout, ljk, nx, ny, jk, g):
    """
    Computes the inside and outside conserved variables function for the third vector component

    Parameters
    ----------
        Hmji   : Water depth at mid-point side i, [3, nElem]
        HUmji  : Flux along-X at mid-point side i, [3, nElem]
        HVmji  : Flux along-Y at mid-point side i, [3, nElem]
        a_in   : Inwards one-sides propagation velocity, [3, nElem]
        a_out  : Outwards one-sides propagation velocity, [3, nElem]
        vmji   : Velocity along the Y-direction at mid-point side i, [3, nElem]
        nx     : Unit-Normal component over X, [3, nElem]
        ny     : Unit-Normal component over Y, [3, nElem]
        ljk    : Side lengths for each triangle, [3, nElem]
        jk     : Global indeces for side neighbor cells quantities, [1, 3*nElem]
        g      : Gravity constant
        choice : 0 at position [i,j] if Xmj1[i,j] is located at the same coordinates as Xmj1.take(jk)[i,j] and 1 otherwise, [3, nElem]
    
    Returns
    -------
        Uin   : Inward mid-point values of the Flux Function, [1, nElem]
        Uout  : Outwards mid-point values of the Flux Function, [1, nElem]
        FUin  : Inside mid-point values of the Flux Function, [1, nElem]
        FUout : Outside mid-point values of the Flux Function, [1, nElem]
        GUin  : Inside mid-point values of the Flux Function, [1, nElem]
        GUout : Outside mid-point values of the Flux Function, [1, nElem]
    """
    #Adjacent state variable values:
    Hin  = HVmj
    Hout = HVmjout
    Coeff = ljk*a_in*a_out/(a_in + a_out)

    Uin  = Coeff[0,:]*Hin[0,:]  + Coeff[1,:]*Hin[1,:]  + Coeff[2,:]*Hin[2,:]
    Uout = Coeff[0,:]*Hout[0,:] + Coeff[1,:]*Hout[1,:] + Coeff[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HUmj*vmj
    Hout = HUmjout*vmjout
    Coeff = ljk*nx/(a_in + a_out)

    FUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    FUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    #Adjacent state variable values:
    Hin  = HVmj*vmj + 0.5*g*Hmj**2
    Hout = HVmjout*vmjout+0.5*g*Hmjout**2
    Coeff = ljk*ny/(a_in + a_out)

    GUin  = Coeff[0,:]*a_out[0,:]*Hin[0,:] + Coeff[1,:]*a_out[1,:]*Hin[1,:] + Coeff[2,:]*a_out[2,:]*Hin[2,:]
    GUout = Coeff[0,:]*a_in[0,:]*Hout[0,:] + Coeff[1,:]*a_in[1,:]*Hout[1,:] + Coeff[2,:]*a_in[2,:]*Hout[2,:]

    return Uin, Uout, FUin, FUout, GUin, GUout

def flux_castro(mesh,Hmj,HUmj,HVmj,umj,vmj,xjk,yjk,cm):

    g=mesh["Constants"]["Gravity"]
    Bx = -mesh["Bx"]
    By = -mesh["By"]

    F1i0 = np.array([HUmj[0,:], HUmj[0,:]*umj[0,:]+0.5*g*Hmj[0,:]**2, HUmj[0,:]*vmj[0,:]])
    F2i0 = np.array([HVmj[0,:], HVmj[0,:]*umj[0,:], HVmj[0,:]*vmj[0,:]+0.5*g*Hmj[0,:]**2])
    F1i1 = np.array([HUmj[1,:], HUmj[1,:]*umj[1,:]+0.5*g*Hmj[1,:]**2, HUmj[1,:]*vmj[1,:]])
    F2i1 = np.array([HVmj[1,:], HVmj[1,:]*umj[1,:], HVmj[1,:]*vmj[1,:]+0.5*g*Hmj[1,:]**2])
    F1i2 = np.array([HUmj[2,:], HUmj[2,:]*umj[2,:]+0.5*g*Hmj[2,:]**2, HUmj[2,:]*vmj[2,:]])
    F2i2 = np.array([HVmj[2,:], HVmj[2,:]*umj[2,:], HVmj[2,:]*vmj[2,:]+0.5*g*Hmj[2,:]**2])

    Fetai0 = mesh["nx"][0,:]*F1i0+mesh["ny"][0,:]*F2i0
    Fetai1 = mesh["nx"][1,:]*F1i1+mesh["ny"][1,:]*F2i1
    Fetai2 = mesh["nx"][2,:]*F1i2+mesh["ny"][2,:]*F2i2

    wi0m = np.array([Hmj[0,:],HUmj[0,:],HVmj[0,:]])
    wi0p = np.array([np.take(Hmj,mesh["jk"])[0,:],np.take(HUmj,mesh["jk"])[0,:],np.take(HVmj,mesh["jk"])[0,:]])
    wi1m = np.array([Hmj[1,:],HUmj[1,:],HVmj[1,:]])
    wi1p = np.array([np.take(Hmj,mesh["jk"])[1,:],np.take(HUmj,mesh["jk"])[1,:],np.take(HVmj,mesh["jk"])[1,:]])
    wi2m = np.array([Hmj[2,:],HUmj[2,:],HVmj[2,:]])
    wi2p = np.array([np.take(Hmj,mesh["jk"])[2,:],np.take(HUmj,mesh["jk"])[2,:],np.take(HVmj,mesh["jk"])[2,:]])

    hi0 = (wi0m[0,:]+wi0p[0,:])/2
    hi1 = (wi1m[0,:]+wi1p[0,:])/2
    hi2 = (wi2m[0,:]+wi2p[0,:])/2

    ui0 = (np.sqrt(wi0m[0,:])*umj[0,:]+np.sqrt(wi0p[0,:])*np.take(umj,mesh["jk"])[0,:])/(np.sqrt(wi0m[0,:])+np.sqrt(wi0p[0,:]))
    ui1 = (np.sqrt(wi1m[0,:])*umj[1,:]+np.sqrt(wi1p[0,:])*np.take(umj,mesh["jk"])[1,:])/(np.sqrt(wi1m[0,:])+np.sqrt(wi1p[0,:]))
    ui2 = (np.sqrt(wi0m[0,:])*umj[2,:]+np.sqrt(wi2p[0,:])*np.take(umj,mesh["jk"])[2,:])/(np.sqrt(wi2m[0,:])+np.sqrt(wi2p[0,:]))

    vi0 = (np.sqrt(wi0m[0,:])*vmj[0,:]+np.sqrt(wi0p[0,:])*np.take(vmj,mesh["jk"])[0,:])/(np.sqrt(wi0m[0,:])+np.sqrt(wi0p[0,:]))
    vi1 = (np.sqrt(wi1m[0,:])*vmj[1,:]+np.sqrt(wi1p[0,:])*np.take(vmj,mesh["jk"])[1,:])/(np.sqrt(wi1m[0,:])+np.sqrt(wi1p[0,:]))
    vi2 = (np.sqrt(wi0m[0,:])*vmj[2,:]+np.sqrt(wi2p[0,:])*np.take(vmj,mesh["jk"])[2,:])/(np.sqrt(wi2m[0,:])+np.sqrt(wi2p[0,:]))

    ci0 = np.sqrt(g*hi0)
    ci1 = np.sqrt(g*hi1)
    ci2 = np.sqrt(g*hi2)

    Ai0 = np.vstack((np.hstack((np.zeros_like(mesh["nx"][0,:])[None,None,:],mesh["nx"][0,:][None,None,:],mesh["ny"][0,:][None,None,:])),np.hstack((((-ui0**2+ci0**2)*mesh["nx"][0,:]-ui0*vi0*mesh["ny"][0,:])[None,None,:],(2*ui0*mesh["nx"][0,:]+vi0*mesh["ny"][0,:])[None,None,:],(ui0*mesh["ny"][0,:])[None,None,:])),np.hstack(((-ui0*vi0*mesh["nx"][0,:]+(-vi0**2+ci0**2)*mesh["ny"][0,:])[None,None,:],(vi0*mesh["nx"][0,:])[None,None,:],(ui0*mesh["nx"][0,:]+2*vi0*mesh["ny"][0,:])[None,None,:])))).T.transpose((0,2,1))
    Ai1 = np.vstack((np.hstack((np.zeros_like(mesh["nx"][1,:])[None,None,:],mesh["nx"][1,:][None,None,:],mesh["ny"][1,:][None,None,:])),np.hstack((((-ui1**2+ci1**2)*mesh["nx"][1,:]-ui1*vi1*mesh["ny"][1,:])[None,None,:],(2*ui1*mesh["nx"][1,:]+vi1*mesh["ny"][1,:])[None,None,:],(ui1*mesh["ny"][1,:])[None,None,:])),np.hstack(((-ui1*vi1*mesh["nx"][1,:]+(-vi1**2+ci1**2)*mesh["ny"][1,:])[None,None,:],(vi1*mesh["nx"][1,:])[None,None,:],(ui1*mesh["nx"][1,:]+2*vi1*mesh["ny"][1,:])[None,None,:])))).T.transpose((0,2,1))
    Ai2 = np.vstack((np.hstack((np.zeros_like(mesh["nx"][2,:])[None,None,:],mesh["nx"][2,:][None,None,:],mesh["ny"][2,:][None,None,:])),np.hstack((((-ui2**2+ci2**2)*mesh["nx"][2,:]-ui2*vi2*mesh["ny"][2,:])[None,None,:],(2*ui2*mesh["nx"][2,:]+vi2*mesh["ny"][2,:])[None,None,:],(ui2*mesh["ny"][2,:])[None,None,:])),np.hstack(((-ui2*vi2*mesh["nx"][2,:]+(-vi2**2+ci2**2)*mesh["ny"][2,:])[None,None,:],(vi2*mesh["nx"][2,:])[None,None,:],(ui2*mesh["nx"][2,:]+2*vi2*mesh["ny"][2,:])[None,None,:])))).T.transpose((0,2,1))

    Si0 = np.array([np.zeros_like(mesh["nx"][0,:]),g*hi0*mesh["nx"][0,:],g*hi0*mesh["ny"][0,:]])
    Si1 = np.array([np.zeros_like(mesh["nx"][0,:]),g*hi1*mesh["nx"][1,:],g*hi1*mesh["ny"][1,:]])
    Si2 = np.array([np.zeros_like(mesh["nx"][0,:]),g*hi2*mesh["nx"][2,:],g*hi2*mesh["ny"][2,:]])

    Hi0m = -mesh["Bmj"][0,:]
    Hi0p = -mesh["Bmj"].take(mesh["jk"])[0,:]
    Hi1m = -mesh["Bmj"][1,:]
    Hi1p = -mesh["Bmj"].take(mesh["jk"])[1,:]
    Hi2m = -mesh["Bmj"][2,:]
    Hi2p = -mesh["Bmj"].take(mesh["jk"])[2,:]

    #Smi0 = np.min(np.array([mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0,mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0+ci0,mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0-ci0]),axis=0)
    #SMi0 = np.max(np.array([mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0,mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0+ci0,mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0-ci0]),axis=0)
    #Smi1 = np.min(np.array([mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1,mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1+ci1,mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1-ci1]),axis=0)
    #SMi1 = np.max(np.array([mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1,mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1+ci1,mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1-ci1]),axis=0)
    #Smi2 = np.min(np.array([mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2,mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2+ci2,mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2-ci2]),axis=0)
    #SMi2 = np.max(np.array([mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2,mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2+ci2,mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2-ci2]),axis=0)

    lambdasi0 = np.array([mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0-ci0,mesh["nx"][0,:]*ui0+mesh["ny"][0,:]*vi0+ci0])
    lambdasi1 = np.array([mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1-ci1,mesh["nx"][1,:]*ui1+mesh["ny"][1,:]*vi1+ci1])
    lambdasi2 = np.array([mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2-ci2,mesh["nx"][2,:]*ui2+mesh["ny"][2,:]*vi2+ci2])

    choicesi0 = np.where(np.abs(lambdasi0[0,:])>=np.abs(lambdasi0[1,:]),0,1)
    choicesi1 = np.where(np.abs(lambdasi1[0,:])>=np.abs(lambdasi1[1,:]),0,1)
    choicesi2 = np.where(np.abs(lambdasi2[0,:])>=np.abs(lambdasi2[1,:]),0,1)

    Smi0 = np.choose(1-choicesi0,lambdasi0)
    SMi0 = np.choose(choicesi0,lambdasi0)
    Smi1 = np.choose(1-choicesi1,lambdasi1)
    SMi1 = np.choose(choicesi1,lambdasi1)
    Smi2 = np.choose(1-choicesi2,lambdasi2)
    SMi2 = np.choose(choicesi2,lambdasi2)

    alpha0i0 = (1-cm)*((SMi0**2)*Smi0*(np.sign(Smi0)-np.sign(SMi0)))/(Smi0-SMi0)**2
    alpha1i0 = (1-cm)*(SMi0*(np.abs(SMi0)-np.abs(Smi0))+Smi0*(np.sign(SMi0)*Smi0 - SMi0*np.sign(Smi0)))/(Smi0-SMi0)**2
    alpha2i0 = (1-cm)*Smi0*(np.sign(Smi0)-np.sign(SMi0))/(Smi0-SMi0)**2
    alpha0i1 = (1-cm)*((SMi1**2)*Smi1*(np.sign(Smi1)-np.sign(SMi1)))/(Smi1-SMi1)**2
    alpha1i1 = (1-cm)*(SMi1*(np.abs(SMi1)-np.abs(Smi1))+Smi1*(np.sign(SMi1)*Smi1 - SMi1*np.sign(Smi1)))/(Smi1-SMi1)**2
    alpha2i1 = (1-cm)*Smi1*(np.sign(Smi1)-np.sign(SMi1))/(Smi1-SMi1)**2
    alpha0i2 = (1-cm)*((SMi2**2)*Smi2*(np.sign(Smi2)-np.sign(SMi2)))/(Smi2-SMi2)**2
    alpha1i2 = (1-cm)*(SMi2*(np.abs(SMi2)-np.abs(Smi2))+Smi2*(np.sign(SMi2)*Smi2 - SMi2*np.sign(Smi2)))/(Smi2-SMi2)**2
    alpha2i2 = (1-cm)*Smi2*(np.sign(Smi2)-np.sign(SMi2))/(Smi2-SMi2)**2

    ones = np.tile(np.array(((1,0,0),(0,1,0),(0,0,1))),(len(Ai0),1,1))

    Pi0 = (ones-alpha0i0[:,None,None]*np.linalg.inv(Ai0)-alpha1i0[:,None,None]*ones-alpha2i0[:,None,None]*Ai0)/2
    Pi0[np.any(np.any(np.isnan(Pi0),axis=-1),axis=-1)]=np.zeros_like(ones[0])

    Pi1 = (ones-alpha0i1[:,None,None]*np.linalg.inv(Ai1)-alpha1i1[:,None,None]*ones-alpha2i1[:,None,None]*Ai1)/2
    Pi1[np.any(np.any(np.isnan(Pi1),axis=-1),axis=-1)]=np.zeros_like(ones[0])

    Pi2 = (ones-alpha0i2[:,None,None]*np.linalg.inv(Ai2)-alpha1i2[:,None,None]*ones-alpha2i2[:,None,None]*Ai2)/2
    Pi2[np.any(np.any(np.isnan(Pi2),axis=-1),axis=-1)]=np.zeros_like(ones[0])

    ch0 = mesh["ljk"][0,:][:,None]*(Fetai0.T+np.einsum('ijk,ik->ij',Pi0,(np.einsum('ijk,ik->ij',Ai0,(wi0p-wi0m).T)-(Hi0p-Hi0m)[:,None]*Si0.T)))
    ch1 = mesh["ljk"][1,:][:,None]*(Fetai1.T+np.einsum('ijk,ik->ij',Pi1,(np.einsum('ijk,ik->ij',Ai1,(wi1p-wi1m).T)-(Hi1p-Hi1m)[:,None]*Si1.T)))
    ch2 = mesh["ljk"][2,:][:,None]*(Fetai2.T+np.einsum('ijk,ik->ij',Pi2,(np.einsum('ijk,ik->ij',Ai2,(wi2p-wi2m).T)-(Hi2p-Hi2m)[:,None]*Si2.T)))

    vint = mesh["Tj"]*(np.array([np.zeros_like(mesh["Tj"]),g*((Hmj[0,:]+Hmj[1,:]+Hmj[2,:])/3)*Bx,g*((Hmj[0,:]+Hmj[1,:]+Hmj[2,:])/3)*By]))

    ch = ch0.T+ch1.T+ch2.T-vint

    return ch

def flux_castro_2(mesh,Hmj,HUmj,HVmj,umj,vmj,i,cm=0):

    cfl = mesh["Constants"]["CFL"]

    g = mesh["Constants"]["Gravity"]

    jk = mesh["jk"]

    etaijx = mesh["nx"][i,:]
    etaijy = mesh["ny"][i,:]

    hi = Hmj[i,:]
    hj = Hmj.take(jk)[i,:]

    qxi = HUmj[i,:]
    qyi = HVmj[i,:]

    qxj = HUmj.take(jk)[i,:]
    qyj = HVmj.take(jk)[i,:]

    uxi = umj[i,:]
    uyi = vmj[i,:]

    uxj = umj.take(jk)[i,:]
    uyj = vmj.take(jk)[i,:]

    Hi = mesh["Bmj"][i,:]
    Hj = mesh["Bmj"].take(jk)[i,:]

    qetaiji   = etaijx*qxi+etaijy*qyi
    qetaijj   = etaijx*qxj+etaijy*qyj

    Sij=np.array([np.zeros_like(hi),g*(hi+hj)/2])*(Hj-Hi)
    FUetaijj = np.array([qetaijj,qetaijj**2/hj+0.5*g*hj**2])
    FUetaiji = np.array([qetaiji,qetaiji**2/hi+0.5*g*hi**2])

    ci = np.sqrt(g*hi)
    cj = np.sqrt(g*hj)

    uetaiji = uxi*etaijx+uyi*etaijy
    uetaijj = uxj*etaijx+uyj*etaijy

    lambdamini = uetaiji-ci
    lambdamaxj = uetaijj+cj

    cij = np.sqrt(g*0.5*(hi+hj))
    uij = (uetaiji*np.sqrt(hi)+uetaijj*np.sqrt(hj))/(np.sqrt(hi)+np.sqrt(hj))

    lambdaminij = uij-cij
    lambdamaxij = uij+cij

    SLij = np.min(np.array([lambdamini,lambdaminij]),axis=0)
    SRij = np.max(np.array([lambdamaxj,lambdamaxij]),axis=0)

    RSij = FUetaijj-FUetaiji-Sij
    RSij[1,:]*=cm

    alphaij0=(SRij*np.abs(SLij)-SLij*np.abs(SRij))/(SRij-SLij)
    alphaij1=(np.abs(SRij)-np.abs(SLij))/(SRij-SLij)

    DUij = np.array([hj-Hj-(hi-Hi),qetaijj-qetaiji])

    Dijm = 0.5*(RSij-(alphaij0*DUij+alphaij1*RSij))
    Fcij = np.array([qetaiji,qetaiji**2/hi])

    phietaijm = Dijm+Fcij

    uetaijtan = np.where(phietaijm[0,:]>0,-uxi*etaijy+uyi*etaijx,-uxj*etaijy+uyj*etaijx)

    phietaijmtan = phietaijm[0,:]*uetaijtan

    Fijm = np.array([phietaijm[0,:],phietaijm[1,:],phietaijmtan])

    FijHLLm = np.array([phietaijm[0,:], phietaijm[1,:]*etaijx-phietaijmtan*etaijy,phietaijmtan*etaijx+phietaijm[1,:]*etaijy])

    lambdaijmax = np.max(np.array([np.abs(SLij),np.abs(SRij)]),axis=0)

    Zij = mesh["ljk"][i,:]*lambdaijmax

    delta_t = np.min(2*cfl*mesh["Tj"]/Zij)

    return FijHLLm, delta_t