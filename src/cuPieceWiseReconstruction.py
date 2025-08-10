import cupy as np

def weno(Zj,T,Ix,Iy,Ixy,x0,y0,coefs,lcoefs,neighs,neighs2,bad_cells):

    o=bad_cells

    lc12 = lcoefs["12"]
    lc23 = lcoefs["23"]
    lc31 = lcoefs["31"]
    lc111= lcoefs["111"]
    lc112= lcoefs["112"]
    lc221= lcoefs["221"]
    lc222= lcoefs["222"]
    lc331= lcoefs["331"]
    lc332= lcoefs["332"]

    U0 = Zj
    U1 = Zj[neighs[0,:]]
    U2 = Zj[neighs[1,:]]
    U3 = Zj[neighs[2,:]]
    U4 = Zj[neighs2[:,0,0]]
    U5 = Zj[neighs2[:,0,1]]
    U6 = Zj[neighs2[:,1,0]]
    U7 = Zj[neighs2[:,1,1]]
    U8 = Zj[neighs2[:,2,0]]
    U9 = Zj[neighs2[:,2,1]]

    #compute least squares quadratic reconstruction
    #Polynomial of the form U_q(x,y) = lsqq0 + lsqq1 (x-x0) + lsqq2 (y-y0) + lsqq3 (x^2-Iy/T) + lsqq4 (y^2-Ix/T) + lsqq5 (xy-Ixy/T)

    lsqq0 = U0
    lsqq1 = coefs[:,0,0]*(U1-U0)+coefs[:,0,1]*(U2-U0)+coefs[:,0,2]*(U3-U0)+coefs[:,0,3]*(U4-U0)+coefs[:,0,4]*(U5-U0)+coefs[:,0,5]*(U6-U0)+coefs[:,0,6]*(U7-U0)+coefs[:,0,7]*(U8-U0)+coefs[:,0,8]*(U9-U0)
    lsqq2 = coefs[:,1,0]*(U1-U0)+coefs[:,1,1]*(U2-U0)+coefs[:,1,2]*(U3-U0)+coefs[:,1,3]*(U4-U0)+coefs[:,1,4]*(U5-U0)+coefs[:,1,5]*(U6-U0)+coefs[:,1,6]*(U7-U0)+coefs[:,1,7]*(U8-U0)+coefs[:,1,8]*(U9-U0)
    lsqq3 = coefs[:,2,0]*(U1-U0)+coefs[:,2,1]*(U2-U0)+coefs[:,2,2]*(U3-U0)+coefs[:,2,3]*(U4-U0)+coefs[:,2,4]*(U5-U0)+coefs[:,2,5]*(U6-U0)+coefs[:,2,6]*(U7-U0)+coefs[:,2,7]*(U8-U0)+coefs[:,2,8]*(U9-U0)
    lsqq4 = coefs[:,3,0]*(U1-U0)+coefs[:,3,1]*(U2-U0)+coefs[:,3,2]*(U3-U0)+coefs[:,3,3]*(U4-U0)+coefs[:,3,4]*(U5-U0)+coefs[:,3,5]*(U6-U0)+coefs[:,3,6]*(U7-U0)+coefs[:,3,7]*(U8-U0)+coefs[:,3,8]*(U9-U0)
    lsqq5 = coefs[:,4,0]*(U1-U0)+coefs[:,4,1]*(U2-U0)+coefs[:,4,2]*(U3-U0)+coefs[:,4,3]*(U4-U0)+coefs[:,4,4]*(U5-U0)+coefs[:,4,5]*(U6-U0)+coefs[:,4,6]*(U7-U0)+coefs[:,4,7]*(U8-U0)+coefs[:,4,8]*(U9-U0)

    ur3 = lambda x,y: lsqq0 + lsqq1*(x-x0) + lsqq2*(y-y0) + lsqq3*(x**2-Iy/T) + lsqq4*(y**2-Ix/T) + lsqq5*(x*y-Ixy/T)

    #compute linear reconstruction for each pair of (consecutive) neighbors
    #Polynomial of the form U_l(x,y) = linAB0 + linAB1 x + linAB2 y, where AB indicates the neighbors considered for that reconstruction

    lin120 = U0
    lin121 = lc12[:,0,0]*(U1-U0)+lc12[:,0,1]*(U2-U0)
    lin122 = lc12[:,1,0]*(U1-U0)+lc12[:,1,1]*(U2-U0)

    ur21 = lambda x,y: lin120+lin121*(x-x0)+lin122*(y-y0)

    lin230 = U0
    lin231 = lc23[:,0,0]*(U2-U0)+lc23[:,0,1]*(U3-U0)
    lin232 = lc23[:,1,0]*(U2-U0)+lc23[:,1,1]*(U3-U0)

    ur22 = lambda x,y: lin230+lin231*(x-x0)+lin232*(y-y0)

    lin310 = U0
    lin311 = lc31[:,0,0]*(U3-U0)+lc31[:,0,1]*(U1-U0)
    lin312 = lc31[:,1,0]*(U3-U0)+lc31[:,1,1]*(U1-U0)

    ur23 = lambda x,y: lin310+lin311*(x-x0)+lin312*(y-y0)

    lin1110 = U0
    lin1111 = lc111[:,0,0]*(U1-U0)+lc111[:,0,1]*(U4-U0)
    lin1112 = lc111[:,1,0]*(U1-U0)+lc111[:,1,1]*(U4-U0)

    ur211 = lambda x,y: lin1110+lin1111*(x-x0)+lin1112*(y-y0)

    lin1120 = U0
    lin1121 = lc112[:,1,1]*(U1-U0)+lc112[:,1,2]*(U5-U0)
    lin1122 = lc112[:,2,1]*(U1-U0)+lc112[:,2,2]*(U5-U0)

    ur212 = lambda x,y: lin1120+lin1121*(x-x0)+lin1122*(y-y0)

    lin2210 = U0
    lin2211 = lc221[:,1,0]*U0+lc221[:,1,1]*U2+lc221[:,1,2]*U6
    lin2212 = lc221[:,2,0]*U0+lc221[:,2,1]*U2+lc221[:,2,2]*U6

    ur221 = lambda x,y: lin2210+lin2211*(x-x0)+lin2212*(y-y0)

    lin2220 = U0
    lin2221 = lc222[:,1,0]*U0+lc222[:,1,1]*U2+lc222[:,1,2]*U7
    lin2222 = lc222[:,2,0]*U0+lc222[:,2,1]*U2+lc222[:,2,2]*U7

    ur222 = lambda x,y: lin2220+lin2221*(x-x0)+lin2222*(y-y0)

    lin3310 = U0
    lin3311 = lc331[:,1,0]*U0+lc331[:,1,1]*U3+lc331[:,1,2]*U8
    lin3312 = lc331[:,2,0]*U0+lc331[:,2,1]*U3+lc331[:,2,2]*U8

    ur231 = lambda x,y: lin3310+lin3311*(x-x0)+lin3312*(y-y0)

    lin3320 = U0
    lin3321 = lc332[:,1,0]*U0+lc332[:,1,1]*U3+lc332[:,1,2]*U9
    lin3322 = lc332[:,2,0]*U0+lc332[:,2,1]*U3+lc332[:,2,2]*U9

    ur232 = lambda x,y: lin3320+lin3321*(x-x0)+lin3322*(y-y0)

    #compute linear weights, smoothness indicator, and nonlinear weights
    #THERE ARE MORE SOPHISTICATED WAYS TO CALCULATE LINEAR WEIGHTS!!!!

    #Linear weights
    alpha = 0.85
    n = 3

    gammar3  = alpha
    gammar21 = (1/(2*n))*(1-alpha)
    gammar22 = (1/(2*n))*(1-alpha)
    gammar23 = (1/(2*n))*(1-alpha)
    gammar211= (1/(4*n))*(1-alpha)
    gammar212= (1/(4*n))*(1-alpha)
    gammar221= (1/(4*n))*(1-alpha)
    gammar222= (1/(4*n))*(1-alpha)
    gammar231= (1/(4*n))*(1-alpha)
    gammar232= (1/(4*n))*(1-alpha)

    #Smoothness indicators
    #Calculated as the sum of the surface integrals of the square of each order k derivative (including mixed derivatives) of the polynomial on the triangle, times the (k-1)-th power of the area of the triangle, for k=1 up to the order of the polynomial
    ISr3  = T*lsqq1**2+4*Iy*lsqq3**2+Ix*lsqq5**2+4*lsqq1*lsqq3*x0*T+2*lsqq1*lsqq5*y0*T+4*lsqq3*lsqq5*Ixy+T*lsqq3**2+4*Ix*lsqq4**2+Iy*lsqq5**2+4*lsqq2*lsqq5*y0*T+2*lsqq2*lsqq5*x0*T+4*lsqq4*lsqq5*Ixy+4*(lsqq3**2)*T**2+4*(lsqq4**2)*T**2+(lsqq5**2)*T**2
    ISr21 = T*lin121**2+T*lin122**2
    ISr22 = T*lin231**2+T*lin232**2
    ISr23 = T*lin311**2+T*lin312**2
    ISr211= T*lin1111**2+T*lin1112**2
    ISr212= T*lin1121**2+T*lin1122**2
    ISr221= T*lin2211**2+T*lin2212**2
    ISr222= T*lin2221**2+T*lin2222**2
    ISr231= T*lin3311**2+T*lin3312**2
    ISr232= T*lin3321**2+T*lin3322**2

    #WENO-Z procedure for nonlinear weights
    tau = (np.abs(ISr3-ISr21)+np.abs(ISr3-ISr22)+np.abs(ISr3-ISr23)+np.abs(ISr3-ISr211)+np.abs(ISr3-ISr212)+np.abs(ISr3-ISr221)+np.abs(ISr3-ISr222)+np.abs(ISr3-ISr231)+np.abs(ISr3-ISr232))/(3*n)

    wr3  = gammar3*(1+(tau**2)/(ISr3+1e-10))
    wr21 = gammar21*(1+(tau**2)/(ISr21+1e-10))
    wr22 = gammar22*(1+(tau**2)/(ISr22+1e-10))
    wr23 = gammar23*(1+(tau**2)/(ISr23+1e-10))
    wr211= gammar211*(1+(tau**2)/(ISr211+1e-10))
    wr212= gammar212*(1+(tau**2)/(ISr212+1e-10))
    wr221= gammar221*(1+(tau**2)/(ISr221+1e-10))
    wr222= gammar222*(1+(tau**2)/(ISr222+1e-10))
    wr231= gammar231*(1+(tau**2)/(ISr231+1e-10))
    wr232= gammar232*(1+(tau**2)/(ISr232+1e-10))

    wr3p=wr3/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr21p=wr21/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr22p=wr22/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr23p=wr23/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr211p=wr211/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr212p=wr212/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr221p=wr221/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr222p=wr222/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr231p=wr231/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)
    wr232p=wr232/(wr3+wr21+wr22+wr23+wr211+wr212+wr221+wr222+wr231+wr232)

    #minmod reconstruction for cells with no second neighbor

    Lxs = np.array([lin121,lin231,lin311])
    Lys = np.array([lin122,lin232,lin312])
    J = np.argmin(Lxs**2 + Lys**2, axis=0)
    Lx = np.choose(J, Lxs)
    Ly = np.choose(J, Lys)
    Lx = np.where((np.abs(Lx)+np.abs(Ly)==np.inf)|(np.isnan(np.abs(Lx)+np.abs(Ly))),0,Lx)
    Ly = np.where((np.abs(Lx)+np.abs(Ly)==np.inf)|(np.isnan(np.abs(Lx)+np.abs(Ly))),0,Ly)

    mm = lambda x,y,ov: (U0+Lx*(x-x0)+Ly*(y-y0))*(1.*o)
    wxm = lambda x,y,ov: Lx*(1.*o)
    wym = lambda x,y,ov: Ly*(1.*o)

    #compute final reconstruction polynomial

    recon = lambda x,y,ov: ((wr3p/gammar3)*(ur3(x,y)-(gammar21*ur21(x,y)+gammar22*ur22(x,y)+gammar23*ur23(x,y)+gammar211*ur211(x,y)+gammar212*ur212(x,y)+gammar221*ur221(x,y)+gammar222*ur222(x,y)+gammar231*ur231(x,y)+gammar232*ur232(x,y))) + wr21p*ur21(x,y)+wr22p*ur22(x,y)+wr23p*ur23(x,y)+wr211p*ur211(x,y)+wr212p*ur212(x,y)+wr221p*ur221(x,y)+wr222p*ur222(x,y)+wr231p*ur231(x,y)+wr232p*ur232(x,y))*(1-1.*o)+mm(x,y,o)

    wx = lambda x,y,ov: ((wr3p/gammar3)*(lsqq1+2*lsqq3*x+lsqq5*y-(gammar21*lin121+gammar22*lin231+gammar23*lin311+gammar211*lin1111+gammar212*lin1121+gammar221*lin2211+gammar222*lin2221+gammar231*lin3311+gammar232*lin3321)) + (wr21p*lin121+wr22p*lin231+wr23p*lin311+wr211p*lin1111+wr212p*lin1121+wr221p*lin2211+wr222p*lin2221+wr231p*lin3311+wr232p*lin3321))*(1-1.*o)+wxm(x,y,o)
    wy = lambda x,y,ov: ((wr3p/gammar3)*(lsqq2+2*lsqq4*y+lsqq5*x-(gammar21*lin122+gammar22*lin232+gammar23*lin312+gammar211*lin1112+gammar212*lin1122+gammar221*lin2212+gammar222*lin2222+gammar231*lin3312+gammar232*lin3322)) + (wr21p*lin122+wr22p*lin232+wr23p*lin312+wr211p*lin1112+wr212p*lin1122+wr221p*lin2212+wr222p*lin2222+wr231p*lin3312+wr232p*lin3322))*(1-1.*o)+wym(x,y,o)

    return recon, wx, wy

def weno2(Zj,T,Ix,Iy,Ixy,x0,y0,xjk,yjk,coefs,lcoefs,neighs,neighs2,bad_cells,tol):

    Zijk = Zj[neighs]

    o=bad_cells

    lc1 = lcoefs["T1"]
    lc2 = lcoefs["T2"]
    lc3 = lcoefs["T3"]
    lc4 = lcoefs["T4"]

    U0 = Zj
    U1 = Zj[neighs[0,:]]
    U2 = Zj[neighs[1,:]]
    U3 = Zj[neighs[2,:]]
    U4 = Zj[neighs2[:,0,0]]
    U5 = Zj[neighs2[:,0,1]]
    U6 = Zj[neighs2[:,1,0]]
    U7 = Zj[neighs2[:,1,1]]
    U8 = Zj[neighs2[:,2,0]]
    U9 = Zj[neighs2[:,2,1]]

    T0 = T
    T1 = T[neighs[0,:]]
    T2 = T[neighs[1,:]]
    T3 = T[neighs[2,:]]
    T4 = T[neighs2[:,0,0]]
    T5 = T[neighs2[:,0,1]]
    T6 = T[neighs2[:,1,0]]
    T7 = T[neighs2[:,1,1]]
    T8 = T[neighs2[:,2,0]]
    T9 = T[neighs2[:,2,1]]

    #compute least squares quadratic reconstruction
    #Polynomial of the form U_q(x,y) = lsqq0 + lsqq1 (x-x0) + lsqq2 (y-y0) + lsqq3 (x^2-Iy/T) + lsqq4 (y^2-Ix/T) + lsqq5 (xy-Ixy/T)

    lsqq0 = U0
    lsqq1 = coefs[:,0,0]*(U1-U0)+coefs[:,0,1]*(U2-U0)+coefs[:,0,2]*(U3-U0)+coefs[:,0,3]*(U4-U0)+coefs[:,0,4]*(U5-U0)+coefs[:,0,5]*(U6-U0)+coefs[:,0,6]*(U7-U0)+coefs[:,0,7]*(U8-U0)+coefs[:,0,8]*(U9-U0)
    lsqq2 = coefs[:,1,0]*(U1-U0)+coefs[:,1,1]*(U2-U0)+coefs[:,1,2]*(U3-U0)+coefs[:,1,3]*(U4-U0)+coefs[:,1,4]*(U5-U0)+coefs[:,1,5]*(U6-U0)+coefs[:,1,6]*(U7-U0)+coefs[:,1,7]*(U8-U0)+coefs[:,1,8]*(U9-U0)
    lsqq3 = coefs[:,2,0]*(U1-U0)+coefs[:,2,1]*(U2-U0)+coefs[:,2,2]*(U3-U0)+coefs[:,2,3]*(U4-U0)+coefs[:,2,4]*(U5-U0)+coefs[:,2,5]*(U6-U0)+coefs[:,2,6]*(U7-U0)+coefs[:,2,7]*(U8-U0)+coefs[:,2,8]*(U9-U0)
    lsqq4 = coefs[:,3,0]*(U1-U0)+coefs[:,3,1]*(U2-U0)+coefs[:,3,2]*(U3-U0)+coefs[:,3,3]*(U4-U0)+coefs[:,3,4]*(U5-U0)+coefs[:,3,5]*(U6-U0)+coefs[:,3,6]*(U7-U0)+coefs[:,3,7]*(U8-U0)+coefs[:,3,8]*(U9-U0)
    lsqq5 = coefs[:,4,0]*(U1-U0)+coefs[:,4,1]*(U2-U0)+coefs[:,4,2]*(U3-U0)+coefs[:,4,3]*(U4-U0)+coefs[:,4,4]*(U5-U0)+coefs[:,4,5]*(U6-U0)+coefs[:,4,6]*(U7-U0)+coefs[:,4,7]*(U8-U0)+coefs[:,4,8]*(U9-U0)

    Uave0 = (U0*T0+U1*T1+U2*T2+U3*T3+U4*T4+U5*T5+U6*T6+U7*T7+U8*T8+U9*T9)/(T0+T1+T2+T3+T4+T5+T6+T7+T8+T9)+1e-40

    ur3 = lambda x,y: lsqq0 + lsqq1*(x-x0) + lsqq2*(y-y0) + lsqq3*(x**2-Iy/T) + lsqq4*(y**2-Ix/T) + lsqq5*(x*y-Ixy/T)

    #compute linear reconstruction for each pair of (consecutive) neighbors
    #Polynomial of the form U_l(x,y) = lin_l0 + lin_l1 (x-x0) + lin_l2 (y-y0), where l indicates the stencil considered for that reconstruction

    lin10 = U0
    lin11 = lc1[:,0,0]*(U1-U0)+lc1[:,0,1]*(U2-U0)+lc1[:,0,2]*(U3-U0)
    lin12 = lc1[:,1,0]*(U1-U0)+lc1[:,1,1]*(U2-U0)+lc1[:,1,2]*(U3-U0)

    Uave1 = (U0*T0+U1*T1+U2*T2+U3*T3)/(T0+T1+T2+T3)+1e-40

    ur21 = lambda x,y: lin10+lin11*(x-x0)+lin12*(y-y0)

    lin20 = U0
    lin21 = lc2[:,0,0]*(U1-U0)+lc2[:,0,1]*(U4-U0)+lc2[:,0,2]*(U5-U0)
    lin22 = lc2[:,1,0]*(U1-U0)+lc2[:,1,1]*(U4-U0)+lc2[:,1,2]*(U5-U0)

    Uave2 = (U0*T0+U1*T1+U4*T4+U5*T5)/(T0+T1+T4+T5)+1e-40

    ur22 = lambda x,y: lin20+lin21*(x-x0)+lin22*(y-y0)

    lin30 = U0
    lin31 = lc3[:,0,0]*(U2-U0)+lc3[:,0,1]*(U6-U0)+lc3[:,0,2]*(U7-U0)
    lin32 = lc3[:,1,0]*(U2-U0)+lc3[:,1,1]*(U6-U0)+lc3[:,1,2]*(U7-U0)

    Uave3 = (U0*T0+U2*T2+U6*T6+U7*T7)/(T0+T2+T6+T7)+1e-40

    ur23 = lambda x,y: lin30+lin31*(x-x0)+lin32*(y-y0)

    lin40 = U0
    lin41 = lc4[:,0,0]*(U3-U0)+lc4[:,0,1]*(U8-U0)+lc4[:,0,2]*(U9-U0)
    lin42 = lc4[:,1,0]*(U3-U0)+lc4[:,1,1]*(U8-U0)+lc4[:,1,2]*(U9-U0)

    Uave4 = (U0*T0+U3*T3+U8*T8+U9*T9)/(T0+T3+T8+T9)+1e-40

    ur24 = lambda x,y: lin40+lin41*(x-x0)+lin42*(y-y0)

    #compute linear weights, smoothness indicator, and nonlinear weights
    #THERE ARE MORE SOPHISTICATED WAYS TO CALCULATE LINEAR WEIGHTS!!!!

    #Linear weights

    gamma0 = 0.96
    gamma1 = 0.01
    gamma2 = 0.01
    gamma3 = 0.01
    gamma4 = 0.01 

    #Smoothness indicators
    #Calculated as the sum of the surface integrals of the square of each order k derivative (including mixed derivatives) of the polynomial on the triangle, times the (k-1)-th power of the area of the triangle, for k=1 up to the order of the polynomial
    ISr3  = T*lsqq1**2 + 4*(Iy)*lsqq3**2 + (Ix)*lsqq5**2 + 4*lsqq1*lsqq3*x0*T + 2*lsqq1*lsqq5*y0*T + 4*lsqq3*lsqq5*(Ixy) + T*lsqq2**2 + 4*(Ix)*lsqq4**2 + (Iy)*lsqq5**2 + 4*lsqq2*lsqq4*y0*T + 2*lsqq2*lsqq5*x0*T + 4*lsqq4*lsqq5*(Ixy) + 4*(lsqq3**2)*T**2 + 4*(lsqq4**2)*T**2 + (lsqq5**2)*T**2
    ISr21 = T*lin11**2+T*lin12**2
    ISr22 = T*lin21**2+T*lin22**2
    ISr23 = T*lin31**2+T*lin32**2
    ISr24 = T*lin41**2+T*lin42**2

    #WENO-Z procedure for nonlinear weights
    tau = ((np.abs(ISr3-ISr21)+np.abs(ISr3-ISr22)+np.abs(ISr3-ISr23)+np.abs(ISr3-ISr24))/(4))

    # Uave0=1
    # Uave1=1
    # Uave2=1
    # Uave3=1

    wr3  = gamma0*(1+(tau**2)/(ISr3**2+1e-4*Uave0**4))
    wr21 = gamma1*(1+(tau**2)/(ISr21**2+1e-4*Uave1**4))
    wr22 = gamma2*(1+(tau**2)/(ISr22**2+1e-4*Uave2**4))
    wr23 = gamma3*(1+(tau**2)/(ISr23**2+1e-4*Uave3**4))
    wr24 = gamma4*(1+(tau**2)/(ISr24**2+1e-4*Uave4**4))

    wr3p  =  wr3/(wr3+wr21+wr22+wr23+wr24)
    wr21p = wr21/(wr3+wr21+wr22+wr23+wr24)
    wr22p = wr22/(wr3+wr21+wr22+wr23+wr24)
    wr23p = wr23/(wr3+wr21+wr22+wr23+wr24)
    wr24p = wr24/(wr3+wr21+wr22+wr23+wr24)

    #minmod reconstruction for cells with no second neighbor

    Lxs = np.array([lin11,lin21,lin31,lin41])
    Lys = np.array([lin12,lin22,lin32,lin42])
    J = np.argmin(Lxs**2 + Lys**2, axis=0)
    Lx = np.choose(J, Lxs)
    Ly = np.choose(J, Lys)
    Lx = np.where((np.abs(Lx)+np.abs(Ly)==np.inf)|(np.isnan(np.abs(Lx)+np.abs(Ly))),0,Lx)
    Ly = np.where((np.abs(Lx)+np.abs(Ly)==np.inf)|(np.isnan(np.abs(Lx)+np.abs(Ly))),0,Ly)

    xmj = 0.5*(xjk[[0,1,2],:] + xjk[[1,2,0],:])
    ymj = 0.5*(yjk[[0,1,2],:] + yjk[[1,2,0],:])

    U12 = Zj + Lx*(xmj[0,:] - x0) + Ly*(ymj[0,:] - y0)
    U23 = Zj + Lx*(xmj[1,:] - x0) + Ly*(ymj[1,:] - y0)
    U31 = Zj + Lx*(xmj[2,:] - x0) + Ly*(ymj[2,:] - y0)

    J12 = (U12 - np.amax(np.array([Zj,Zijk[0,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[0,:]]) ,axis=0 ) - U12 > tol)
    J23 = (U23 - np.amax(np.array([Zj,Zijk[1,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[1,:]]) ,axis=0 ) - U23 > tol)
    J31 = (U31 - np.amax(np.array([Zj,Zijk[2,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[2,:]]) ,axis=0 ) - U31 > tol)

    J = (J12 | J23 | J31)

    #print(np.count_nonzero(~J))

    Lx[J] = 0.0
    Ly[J] = 0.0

    mm = lambda x,y,ov: (U0+Lx*(x-x0)+Ly*(y-y0))*(1.*o)
    wxm = lambda x,y,ov: Lx*(1.*o)
    wym = lambda x,y,ov: Ly*(1.*o)

    #compute final reconstruction polynomial

    recon = lambda x,y,ov: (wr3p*((1/gamma0)*ur3(x,y)-((gamma1/gamma0)*ur21(x,y)+(gamma2/gamma0)*ur22(x,y)+(gamma3/gamma0)*ur23(x,y)+(gamma4/gamma0)*ur24(x,y)))+wr21p*ur21(x,y)+wr22p*ur22(x,y)+wr23p*ur23(x,y)+wr24p*ur24(x,y))*(1-1.*o)+mm(x,y,ov)

    wx = lambda x,y,ov: ((wr3p/gamma0)*(lsqq1+2*lsqq3*x+lsqq5*y-(gamma1*lin11+gamma2*lin21+gamma3*lin31+gamma4*lin41)) + (wr21p*lin11+wr22p*lin21+wr23p*lin31+wr24p*lin41))*(1-1.*o)+wxm(x,y,ov)
    wy = lambda x,y,ov: ((wr3p/gamma0)*(lsqq2+2*lsqq4*y+lsqq5*x-(gamma1*lin12+gamma2*lin22+gamma3*lin32+gamma4*lin42)) + (wr21p*lin12+wr22p*lin22+wr23p*lin32+wr24p*lin42))*(1-1.*o)+wym(x,y,ov)

    return recon, wx, wy

def minmod2(Zj,stencils,NChoose,rjk,xjk,yjk,xj,yj,Ix,Iy,Ixy,T,TOL):

    tol=TOL

    l=NChoose

    ijk=NChoose[0][:,:,0].T

    rj=np.max(rjk,axis=0)
    
    U0=Zj

    stencil=stencils["n2n"][0]

    a11=stencils["aij"][0][0,0]
    a12=stencils["aij"][0][0,1]
    a13=stencils["aij"][0][0,2]
    a14=stencils["aij"][0][0,3]
    a15=stencils["aij"][0][0,4]
    a21=stencils["aij"][0][1,0]
    a22=stencils["aij"][0][1,1]
    a23=stencils["aij"][0][1,2]
    a24=stencils["aij"][0][1,3]
    a25=stencils["aij"][0][1,4]
    a31=stencils["aij"][0][2,0]
    a32=stencils["aij"][0][2,1]
    a33=stencils["aij"][0][2,2]
    a34=stencils["aij"][0][2,3]
    a35=stencils["aij"][0][2,4]
    a41=stencils["aij"][0][3,0]
    a42=stencils["aij"][0][3,1]
    a43=stencils["aij"][0][3,2]
    a44=stencils["aij"][0][3,3]
    a45=stencils["aij"][0][3,4]
    a51=stencils["aij"][0][4,0]
    a52=stencils["aij"][0][4,1]
    a53=stencils["aij"][0][4,2]
    a54=stencils["aij"][0][4,3]
    a55=stencils["aij"][0][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux1  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy1  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx1 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy1 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy1 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][0]

    Ux1[bool_aux]  = 1e12
    Uy1[bool_aux]  = 1e12
    Uxx1[bool_aux] = 1e12
    Uyy1[bool_aux] = 1e12
    Uxy1[bool_aux] = 1e12

    stencil=stencils["n2n"][1]

    a11=stencils["aij"][1][0,0]
    a12=stencils["aij"][1][0,1]
    a13=stencils["aij"][1][0,2]
    a14=stencils["aij"][1][0,3]
    a15=stencils["aij"][1][0,4]
    a21=stencils["aij"][1][1,0]
    a22=stencils["aij"][1][1,1]
    a23=stencils["aij"][1][1,2]
    a24=stencils["aij"][1][1,3]
    a25=stencils["aij"][1][1,4]
    a31=stencils["aij"][1][2,0]
    a32=stencils["aij"][1][2,1]
    a33=stencils["aij"][1][2,2]
    a34=stencils["aij"][1][2,3]
    a35=stencils["aij"][1][2,4]
    a41=stencils["aij"][1][3,0]
    a42=stencils["aij"][1][3,1]
    a43=stencils["aij"][1][3,2]
    a44=stencils["aij"][1][3,3]
    a45=stencils["aij"][1][3,4]
    a51=stencils["aij"][1][4,0]
    a52=stencils["aij"][1][4,1]
    a53=stencils["aij"][1][4,2]
    a54=stencils["aij"][1][4,3]
    a55=stencils["aij"][1][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]

    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux2  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy2  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx2 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy2 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy2 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][1]

    Ux2[bool_aux]  = 1e12
    Uy2[bool_aux]  = 1e12
    Uxx2[bool_aux] = 1e12
    Uyy2[bool_aux] = 1e12
    Uxy2[bool_aux] = 1e12

    stencil=stencils["n2n"][2]

    a11=stencils["aij"][2][0,0]
    a12=stencils["aij"][2][0,1]
    a13=stencils["aij"][2][0,2]
    a14=stencils["aij"][2][0,3]
    a15=stencils["aij"][2][0,4]
    a21=stencils["aij"][2][1,0]
    a22=stencils["aij"][2][1,1]
    a23=stencils["aij"][2][1,2]
    a24=stencils["aij"][2][1,3]
    a25=stencils["aij"][2][1,4]
    a31=stencils["aij"][2][2,0]
    a32=stencils["aij"][2][2,1]
    a33=stencils["aij"][2][2,2]
    a34=stencils["aij"][2][2,3]
    a35=stencils["aij"][2][2,4]
    a41=stencils["aij"][2][3,0]
    a42=stencils["aij"][2][3,1]
    a43=stencils["aij"][2][3,2]
    a44=stencils["aij"][2][3,3]
    a45=stencils["aij"][2][3,4]
    a51=stencils["aij"][2][4,0]
    a52=stencils["aij"][2][4,1]
    a53=stencils["aij"][2][4,2]
    a54=stencils["aij"][2][4,3]
    a55=stencils["aij"][2][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux3  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy3  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx3 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy3 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy3 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][2]

    Ux3[bool_aux]  = 1e12
    Uy3[bool_aux]  = 1e12
    Uxx3[bool_aux] = 1e12
    Uyy3[bool_aux] = 1e12
    Uxy3[bool_aux] = 1e12

    stencil=stencils["n2n"][3]

    a11=stencils["aij"][3][0,0]
    a12=stencils["aij"][3][0,1]
    a13=stencils["aij"][3][0,2]
    a14=stencils["aij"][3][0,3]
    a15=stencils["aij"][3][0,4]
    a21=stencils["aij"][3][1,0]
    a22=stencils["aij"][3][1,1]
    a23=stencils["aij"][3][1,2]
    a24=stencils["aij"][3][1,3]
    a25=stencils["aij"][3][1,4]
    a31=stencils["aij"][3][2,0]
    a32=stencils["aij"][3][2,1]
    a33=stencils["aij"][3][2,2]
    a34=stencils["aij"][3][2,3]
    a35=stencils["aij"][3][2,4]
    a41=stencils["aij"][3][3,0]
    a42=stencils["aij"][3][3,1]
    a43=stencils["aij"][3][3,2]
    a44=stencils["aij"][3][3,3]
    a45=stencils["aij"][3][3,4]
    a51=stencils["aij"][3][4,0]
    a52=stencils["aij"][3][4,1]
    a53=stencils["aij"][3][4,2]
    a54=stencils["aij"][3][4,3]
    a55=stencils["aij"][3][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux4  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy4  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx4 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy4 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy4 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][3]

    Ux4[bool_aux]  = 1e12
    Uy4[bool_aux]  = 1e12
    Uxx4[bool_aux] = 1e12
    Uyy4[bool_aux] = 1e12
    Uxy4[bool_aux] = 1e12

    stencil=stencils["n2n"][4]

    a11=stencils["aij"][4][0,0]
    a12=stencils["aij"][4][0,1]
    a13=stencils["aij"][4][0,2]
    a14=stencils["aij"][4][0,3]
    a15=stencils["aij"][4][0,4]
    a21=stencils["aij"][4][1,0]
    a22=stencils["aij"][4][1,1]
    a23=stencils["aij"][4][1,2]
    a24=stencils["aij"][4][1,3]
    a25=stencils["aij"][4][1,4]
    a31=stencils["aij"][4][2,0]
    a32=stencils["aij"][4][2,1]
    a33=stencils["aij"][4][2,2]
    a34=stencils["aij"][4][2,3]
    a35=stencils["aij"][4][2,4]
    a41=stencils["aij"][4][3,0]
    a42=stencils["aij"][4][3,1]
    a43=stencils["aij"][4][3,2]
    a44=stencils["aij"][4][3,3]
    a45=stencils["aij"][4][3,4]
    a51=stencils["aij"][4][4,0]
    a52=stencils["aij"][4][4,1]
    a53=stencils["aij"][4][4,2]
    a54=stencils["aij"][4][4,3]
    a55=stencils["aij"][4][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux5  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy5  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx5 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy5 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy5 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][4]

    Ux5[bool_aux]  = 1e12
    Uy5[bool_aux]  = 1e12
    Uxx5[bool_aux] = 1e12
    Uyy5[bool_aux] = 1e12
    Uxy5[bool_aux] = 1e12

    stencil=stencils["n2n"][5]

    a11=stencils["aij"][5][0,0]
    a12=stencils["aij"][5][0,1]
    a13=stencils["aij"][5][0,2]
    a14=stencils["aij"][5][0,3]
    a15=stencils["aij"][5][0,4]
    a21=stencils["aij"][5][1,0]
    a22=stencils["aij"][5][1,1]
    a23=stencils["aij"][5][1,2]
    a24=stencils["aij"][5][1,3]
    a25=stencils["aij"][5][1,4]
    a31=stencils["aij"][5][2,0]
    a32=stencils["aij"][5][2,1]
    a33=stencils["aij"][5][2,2]
    a34=stencils["aij"][5][2,3]
    a35=stencils["aij"][5][2,4]
    a41=stencils["aij"][5][3,0]
    a42=stencils["aij"][5][3,1]
    a43=stencils["aij"][5][3,2]
    a44=stencils["aij"][5][3,3]
    a45=stencils["aij"][5][3,4]
    a51=stencils["aij"][5][4,0]
    a52=stencils["aij"][5][4,1]
    a53=stencils["aij"][5][4,2]
    a54=stencils["aij"][5][4,3]
    a55=stencils["aij"][5][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux6  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy6  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx6 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy6 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy6 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][5]

    Ux6[bool_aux]  = 1e12
    Uy6[bool_aux]  = 1e12
    Uxx6[bool_aux] = 1e12
    Uyy6[bool_aux] = 1e12
    Uxy6[bool_aux] = 1e12

    stencil=stencils["n2n"][6]

    a11=stencils["aij"][6][0,0]
    a12=stencils["aij"][6][0,1]
    a13=stencils["aij"][6][0,2]
    a14=stencils["aij"][6][0,3]
    a15=stencils["aij"][6][0,4]
    a21=stencils["aij"][6][1,0]
    a22=stencils["aij"][6][1,1]
    a23=stencils["aij"][6][1,2]
    a24=stencils["aij"][6][1,3]
    a25=stencils["aij"][6][1,4]
    a31=stencils["aij"][6][2,0]
    a32=stencils["aij"][6][2,1]
    a33=stencils["aij"][6][2,2]
    a34=stencils["aij"][6][2,3]
    a35=stencils["aij"][6][2,4]
    a41=stencils["aij"][6][3,0]
    a42=stencils["aij"][6][3,1]
    a43=stencils["aij"][6][3,2]
    a44=stencils["aij"][6][3,3]
    a45=stencils["aij"][6][3,4]
    a51=stencils["aij"][6][4,0]
    a52=stencils["aij"][6][4,1]
    a53=stencils["aij"][6][4,2]
    a54=stencils["aij"][6][4,3]
    a55=stencils["aij"][6][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux7  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy7  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx7 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy7 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy7 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][6]

    Ux7[bool_aux]  = 1e12
    Uy7[bool_aux]  = 1e12
    Uxx7[bool_aux] = 1e12
    Uyy7[bool_aux] = 1e12
    Uxy7[bool_aux] = 1e12

    stencil=stencils["n2n"][7]

    a11=stencils["aij"][7][0,0]
    a12=stencils["aij"][7][0,1]
    a13=stencils["aij"][7][0,2]
    a14=stencils["aij"][7][0,3]
    a15=stencils["aij"][7][0,4]
    a21=stencils["aij"][7][1,0]
    a22=stencils["aij"][7][1,1]
    a23=stencils["aij"][7][1,2]
    a24=stencils["aij"][7][1,3]
    a25=stencils["aij"][7][1,4]
    a31=stencils["aij"][7][2,0]
    a32=stencils["aij"][7][2,1]
    a33=stencils["aij"][7][2,2]
    a34=stencils["aij"][7][2,3]
    a35=stencils["aij"][7][2,4]
    a41=stencils["aij"][7][3,0]
    a42=stencils["aij"][7][3,1]
    a43=stencils["aij"][7][3,2]
    a44=stencils["aij"][7][3,3]
    a45=stencils["aij"][7][3,4]
    a51=stencils["aij"][7][4,0]
    a52=stencils["aij"][7][4,1]
    a53=stencils["aij"][7][4,2]
    a54=stencils["aij"][7][4,3]
    a55=stencils["aij"][7][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux8  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy8  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx8 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy8 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy8 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][7]

    Ux8[bool_aux]  = 1e12
    Uy8[bool_aux]  = 1e12
    Uxx8[bool_aux] = 1e12
    Uyy8[bool_aux] = 1e12
    Uxy8[bool_aux] = 1e12

    stencil=stencils["n2n"][8]

    a11=stencils["aij"][8][0,0]
    a12=stencils["aij"][8][0,1]
    a13=stencils["aij"][8][0,2]
    a14=stencils["aij"][8][0,3]
    a15=stencils["aij"][8][0,4]
    a21=stencils["aij"][8][1,0]
    a22=stencils["aij"][8][1,1]
    a23=stencils["aij"][8][1,2]
    a24=stencils["aij"][8][1,3]
    a25=stencils["aij"][8][1,4]
    a31=stencils["aij"][8][2,0]
    a32=stencils["aij"][8][2,1]
    a33=stencils["aij"][8][2,2]
    a34=stencils["aij"][8][2,3]
    a35=stencils["aij"][8][2,4]
    a41=stencils["aij"][8][3,0]
    a42=stencils["aij"][8][3,1]
    a43=stencils["aij"][8][3,2]
    a44=stencils["aij"][8][3,3]
    a45=stencils["aij"][8][3,4]
    a51=stencils["aij"][8][4,0]
    a52=stencils["aij"][8][4,1]
    a53=stencils["aij"][8][4,2]
    a54=stencils["aij"][8][4,3]
    a55=stencils["aij"][8][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux9  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy9  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx9 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy9 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy9 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][8]

    Ux9[bool_aux]  = 1e12
    Uy9[bool_aux]  = 1e12
    Uxx9[bool_aux] = 1e12
    Uyy9[bool_aux] = 1e12
    Uxy9[bool_aux] = 1e12

    stencil=stencils["n2n"][9]

    a11=stencils["aij"][9][0,0]
    a12=stencils["aij"][9][0,1]
    a13=stencils["aij"][9][0,2]
    a14=stencils["aij"][9][0,3]
    a15=stencils["aij"][9][0,4]
    a21=stencils["aij"][9][1,0]
    a22=stencils["aij"][9][1,1]
    a23=stencils["aij"][9][1,2]
    a24=stencils["aij"][9][1,3]
    a25=stencils["aij"][9][1,4]
    a31=stencils["aij"][9][2,0]
    a32=stencils["aij"][9][2,1]
    a33=stencils["aij"][9][2,2]
    a34=stencils["aij"][9][2,3]
    a35=stencils["aij"][9][2,4]
    a41=stencils["aij"][9][3,0]
    a42=stencils["aij"][9][3,1]
    a43=stencils["aij"][9][3,2]
    a44=stencils["aij"][9][3,3]
    a45=stencils["aij"][9][3,4]
    a51=stencils["aij"][9][4,0]
    a52=stencils["aij"][9][4,1]
    a53=stencils["aij"][9][4,2]
    a54=stencils["aij"][9][4,3]
    a55=stencils["aij"][9][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux10  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy10  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx10 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy10 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy10 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][9]

    Ux10[bool_aux]  = 1e12
    Uy10[bool_aux]  = 1e12
    Uxx10[bool_aux] = 1e12
    Uyy10[bool_aux] = 1e12
    Uxy10[bool_aux] = 1e12

    stencil=stencils["n2n"][10]

    a11=stencils["aij"][10][0,0]
    a12=stencils["aij"][10][0,1]
    a13=stencils["aij"][10][0,2]
    a14=stencils["aij"][10][0,3]
    a15=stencils["aij"][10][0,4]
    a21=stencils["aij"][10][1,0]
    a22=stencils["aij"][10][1,1]
    a23=stencils["aij"][10][1,2]
    a24=stencils["aij"][10][1,3]
    a25=stencils["aij"][10][1,4]
    a31=stencils["aij"][10][2,0]
    a32=stencils["aij"][10][2,1]
    a33=stencils["aij"][10][2,2]
    a34=stencils["aij"][10][2,3]
    a35=stencils["aij"][10][2,4]
    a41=stencils["aij"][10][3,0]
    a42=stencils["aij"][10][3,1]
    a43=stencils["aij"][10][3,2]
    a44=stencils["aij"][10][3,3]
    a45=stencils["aij"][10][3,4]
    a51=stencils["aij"][10][4,0]
    a52=stencils["aij"][10][4,1]
    a53=stencils["aij"][10][4,2]
    a54=stencils["aij"][10][4,3]
    a55=stencils["aij"][10][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux11  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy11  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx11 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy11 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy11 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][10]

    Ux11[bool_aux]  = 1e12
    Uy11[bool_aux]  = 1e12
    Uxx11[bool_aux] = 1e12
    Uyy11[bool_aux] = 1e12
    Uxy11[bool_aux] = 1e12

    stencil=stencils["n2n"][11]

    a11=stencils["aij"][11][0,0]
    a12=stencils["aij"][11][0,1]
    a13=stencils["aij"][11][0,2]
    a14=stencils["aij"][11][0,3]
    a15=stencils["aij"][11][0,4]
    a21=stencils["aij"][11][1,0]
    a22=stencils["aij"][11][1,1]
    a23=stencils["aij"][11][1,2]
    a24=stencils["aij"][11][1,3]
    a25=stencils["aij"][11][1,4]
    a31=stencils["aij"][11][2,0]
    a32=stencils["aij"][11][2,1]
    a33=stencils["aij"][11][2,2]
    a34=stencils["aij"][11][2,3]
    a35=stencils["aij"][11][2,4]
    a41=stencils["aij"][11][3,0]
    a42=stencils["aij"][11][3,1]
    a43=stencils["aij"][11][3,2]
    a44=stencils["aij"][11][3,3]
    a45=stencils["aij"][11][3,4]
    a51=stencils["aij"][11][4,0]
    a52=stencils["aij"][11][4,1]
    a53=stencils["aij"][11][4,2]
    a54=stencils["aij"][11][4,3]
    a55=stencils["aij"][11][4,4]

    U1=Zj[l[(stencil[0][1]+2)//2]][:,stencil[0][0],stencil[0][1]]
    U2=Zj[l[(stencil[1][1]+2)//2]][:,stencil[1][0],stencil[1][1]]
    U3=Zj[l[(stencil[2][1]+2)//2]][:,stencil[2][0],stencil[2][1]]
    U4=Zj[l[(stencil[3][1]+2)//2]][:,stencil[3][0],stencil[3][1]]
    U5=Zj[l[(stencil[4][1]+2)//2]][:,stencil[4][0],stencil[4][1]]
    
    dif1 = U1-U0
    dif2 = U2-U0
    dif3 = U3-U0
    dif4 = U4-U0
    dif5 = U5-U0

    Ux12  = a11*dif1+a12*dif2+a13*dif3+a14*dif4+a15*dif5
    Uy12  = a21*dif1+a22*dif2+a23*dif3+a24*dif4+a25*dif5
    Uxx12 = a31*dif1+a32*dif2+a33*dif3+a34*dif4+a35*dif5
    Uyy12 = a41*dif1+a42*dif2+a43*dif3+a44*dif4+a45*dif5
    Uxy12 = a51*dif1+a52*dif2+a53*dif3+a54*dif4+a55*dif5

    bool_aux=stencils["bool_aux"][11]

    Ux12[bool_aux]  = 1e12
    Uy12[bool_aux]  = 1e12
    Uxx12[bool_aux] = 1e12
    Uyy12[bool_aux] = 1e12
    Uxy12[bool_aux] = 1e12

    Ux = np.array([Ux1,Ux2,Ux3,Ux4,Ux5,Ux6,Ux7,Ux8,Ux9,Ux10,Ux11,Ux12])
    Uy = np.array([Uy1,Uy2,Uy3,Uy4,Uy5,Uy6,Uy7,Uy8,Uy9,Uy10,Uy11,Uy12])
    Uxx= np.array([Uxx1,Uxx2,Uxx3,Uxx4,Uxx5,Uxx6,Uxx7,Uxx8,Uxx9,Uxx10,Uxx11,Uxx12])
    Uyy= np.array([Uyy1,Uyy2,Uyy3,Uyy4,Uyy5,Uyy6,Uyy7,Uyy8,Uyy9,Uyy10,Uyy11,Uyy12])
    Uxy= np.array([Uxy1,Uxy2,Uxy3,Uxy4,Uxy5,Uxy6,Uxy7,Uxy8,Uxy9,Uxy10,Uxy11,Uxy12])

    Jmin = np.argmin(np.sqrt((Ux+Uxx*rj/3+Uxy)**2 + (Uy+Uyy*rj/3+Uxy)**2), axis=0) #select the position where the super gradient is the lowest
    Lx = np.choose(Jmin, Ux)
    Ly = np.choose(Jmin, Uy)
    Lxx= np.choose(Jmin, Uxx)
    Lyy= np.choose(Jmin, Uyy)
    Lxy= np.choose(Jmin, Uxy)

    #Lx = np.mean(Ux,axis=0)
    #Ly = np.mean(Uy,axis=0)
    #Lxx= np.mean(Uxx,axis=0)
    #Lyy= np.mean(Uyy,axis=0)
    #Lxy= np.mean(Uxy,axis=0)

    # Finding a Local Extrema at Mid-points
    # 
    # Mid-points coordinates for each element
    c = 0.5+np.sqrt(3)/6

    #xmjk1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    #xmjk2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]
    xmjkc = 0.5*xjk[[0,1,2],:]+0.5*xjk[[1,2,0],:]

    #ymjk1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    #ymjk2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]
    ymjkc = 0.5*yjk[[0,1,2],:]+0.5*yjk[[1,2,0],:]

    #U121 = Zj + Lx*(xmjk1[0,:]-xj) + Ly*(ymjk1[0,:]-yj) + 0.5*Lxx*(xmjk1[0,:]-xj)**2 + 0.5*Lyy*(ymjk1[0,:]-yj)**2 + Lxy*(xmjk1[0,:]-xj)*(ymjk1[0,:]-yj)
    U12c = Zj + Lx*(xmjkc[0,:]-xj) + Ly*(ymjkc[0,:]-yj) + 0.5*Lxx*((xmjkc[0,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjkc[0,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjkc[0,:]-xj)*(ymjkc[0,:]-yj)-(Ixy/T-xj*yj))
    #U122 = Zj + Lx*(xmjk2[0,:]-xj) + Ly*(ymjk2[0,:]-yj) + 0.5*Lxx*(xmjk2[0,:]-xj)**2 + 0.5*Lyy*(ymjk2[0,:]-yj)**2 + Lxy*(xmjk2[0,:]-xj)*(ymjk2[0,:]-yj)

    #U231 = Zj + Lx*(xmjk1[1,:]-xj) + Ly*(ymjk1[1,:]-yj) + 0.5*Lxx*(xmjk1[1,:]-xj)**2 + 0.5*Lyy*(ymjk1[1,:]-yj)**2 + Lxy*(xmjk1[1,:]-xj)*(ymjk1[1,:]-yj)
    U23c = Zj + Lx*(xmjkc[1,:]-xj) + Ly*(ymjkc[1,:]-yj) + 0.5*Lxx*((xmjkc[1,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjkc[1,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjkc[1,:]-xj)*(ymjkc[1,:]-yj)-(Ixy/T-xj*yj))
    #U232 = Zj + Lx*(xmjk2[1,:]-xj) + Ly*(ymjk2[1,:]-yj) + 0.5*Lxx*(xmjk2[1,:]-xj)**2 + 0.5*Lyy*(ymjk2[1,:]-yj)**2 + Lxy*(xmjk2[1,:]-xj)*(ymjk2[1,:]-yj)
    
    #U311 = Zj + Lx*(xmjk1[2,:]-xj) + Ly*(ymjk1[2,:]-yj) + 0.5*Lxx*(xmjk1[2,:]-xj)**2 + 0.5*Lyy*(ymjk1[2,:]-yj)**2 + Lxy*(xmjk1[2,:]-xj)*(ymjk1[2,:]-yj)
    U31c = Zj + Lx*(xmjkc[2,:]-xj) + Ly*(ymjkc[2,:]-yj) + 0.5*Lxx*((xmjkc[2,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjkc[2,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjkc[2,:]-xj)*(ymjkc[2,:]-yj)-(Ixy/T-xj*yj))
    #U312 = Zj + Lx*(xmjk2[2,:]-xj) + Ly*(ymjk2[2,:]-yj) + 0.5*Lxx*(xmjk2[2,:]-xj)**2 + 0.5*Lyy*(ymjk2[2,:]-yj)**2 + Lxy*(xmjk2[2,:]-xj)*(ymjk2[2,:]-yj)

    Zijk=Zj[ijk]

    #J121 = (U121 - np.amax(np.array([Zj,Zijk[0,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[0,:]]) ,axis=0 ) - U121 > tol)
    #J231 = (U231 - np.amax(np.array([Zj,Zijk[1,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[1,:]]) ,axis=0 ) - U231 > tol)
    #J311 = (U311 - np.amax(np.array([Zj,Zijk[2,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[2,:]]) ,axis=0 ) - U311 > tol)

    #J122 = (U122 - np.amax(np.array([Zj,Zijk[0,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[0,:]]) ,axis=0 ) - U122 > tol)
    #J232 = (U232 - np.amax(np.array([Zj,Zijk[1,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[1,:]]) ,axis=0 ) - U232 > tol)
    #J312 = (U312 - np.amax(np.array([Zj,Zijk[2,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[2,:]]) ,axis=0 ) - U312 > tol)

    J12c = (U12c - np.amax(np.array([Zj,Zijk[0,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[0,:]]) ,axis=0 ) - U12c > tol)
    J23c = (U23c - np.amax(np.array([Zj,Zijk[1,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[1,:]]) ,axis=0 ) - U23c > tol)
    J31c = (U31c - np.amax(np.array([Zj,Zijk[2,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[2,:]]) ,axis=0 ) - U31c > tol)

    #U12m = (U121+U122)/2
    #U23m = (U231+U232)/2
    #U31m = (U311+U312)/2

    #J12m = (U12m - np.amax(np.array([Zj,Zijk[0,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[0,:]]) ,axis=0 ) - U12m > tol)
    #J23m = (U23m - np.amax(np.array([Zj,Zijk[1,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[1,:]]) ,axis=0 ) - U23m > tol)
    #J31m = (U31m - np.amax(np.array([Zj,Zijk[2,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[2,:]]) ,axis=0 ) - U31m > tol)

    #J = ((J121 & J122) | (J231 & J232) | (J311 & J312))
    J = (J12c | J23c | J31c)
    #J = (J12m | J23m | J31m)

    print(np.count_nonzero(~J))

    Lx[J]  = 0.0
    Ly[J]  = 0.0
    Lxx[J] = 0.0
    Lyy[J] = 0.0
    Lxy[J] = 0.0

    return Lx,Ly,Lxx,Lyy,Lxy

def minmod(Zj, xj, yj, xjk, yjk, ijk,TOL,theta):
    """
    This function calculates the cell gradient according to Bryson et al., 2011
    Bryson, Steve; Epshteyn, Yekaterina; Kurganov, Alexander; Petrova, Guergana. Well-balanced positivity preserving central-upwind scheme on triangular grids for the Saint-Venant system. ESAIM: Mathematical Modelling and Numerical Analysis - Modélisation Mathématique et Analyse Numérique, Volume 45 (2011) no. 3, pp. 423-446. doi : 10.1051/m2an/2010060. http://www.numdam.org/articles/10.1051/m2an/2010060/

    Parameters
    ----------
    Zj  : Cell-Centered values of quantity Z, [1, nElem]
    xj  : Coordinate in x-direction of X, [1, nElem]
    yj  : Coordinate in y-direction of X, [1, nElem]
    xjk : Coordinate in x-direction of the triangle vertices for each element [3, nElem]
    yjk : Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    ijk : Elements neighboors to the element j [3, nElem]
    DRY : Dry cell tolerance
    
    Returns
    -------
    Lx  : Partial derivative of X along x-direction, [1, nElem]
    Ly  : Partial derivative of X along y-direction, [1, nElem]
    """
    xijk = xj[ijk]
    yijk = yj[ijk]
    Zijk = Zj[ijk]

    #COMPUTES GRADIENTS BETWEEN NEIGHBOR ELEMENTS
    
    tol = TOL
    
    #Plane L12:
    D = -((xijk[0,:] - xj)*(yijk[1,:] - yj) - (xijk[1,:] - xj)*(yijk[0,:] - yj))
    bool_aux = ( np.abs(D) < tol )
    D[bool_aux] = 1.0

    Lj12_x = ((yijk[0,:] - yj)*(Zijk[1,:] - Zj) - (yijk[1,:] - yj)*(Zijk[0,:] - Zj))/D
    Lj12_y = ((xijk[1,:] - xj)*(Zijk[0,:] - Zj) - (xijk[0,:] - xj)*(Zijk[1,:] - Zj))/D
    
    Lj12_x[bool_aux] = 1e20
    Lj12_y[bool_aux] = 1e20
        
    #Plane L23:
    D = -((xijk[1,:] - xj)*(yijk[2,:] - yj) - (xijk[2,:] - xj)*(yijk[1,:] - yj))
    bool_aux = ( np.abs(D) < tol )
    D[bool_aux] = 1.0

    Lj23_x = ((yijk[1,:] - yj)*(Zijk[2,:] - Zj) - (yijk[2,:] - yj)*(Zijk[1,:] - Zj))/D
    Lj23_y = ((xijk[2,:] - xj)*(Zijk[1,:] - Zj) - (xijk[1,:] - xj)*(Zijk[2,:] - Zj))/D

    Lj23_x[bool_aux] = 1e20
    Lj23_y[bool_aux] = 1e20
        
    #Plane L31:
    D = -((xijk[2,:] - xj)*(yijk[0,:] - yj) - (xijk[0,:] - xj)*(yijk[2,:] - yj))
    bool_aux = ( np.abs(D) < tol )
    D[bool_aux] = 1.0
    
    Lj31_x = ((yijk[2,:] - yj)*(Zijk[0,:] - Zj) - (yijk[0,:] - yj)*(Zijk[2,:] - Zj))/D
    Lj31_y = ((xijk[0,:] - xj)*(Zijk[2,:] - Zj) - (xijk[2,:] - xj)*(Zijk[0,:] - Zj))/D

    Lj31_x[bool_aux] = 1e20
    Lj31_y[bool_aux] = 1e20
        
    #Smallest magnitude of the Gradient
    Lx = theta*np.array([Lj12_x, Lj23_x, Lj31_x])
    Ly = theta*np.array([Lj12_y, Lj23_y, Lj31_y])

    Jmin = np.argmin(np.sqrt(Lx**2 + Ly**2), axis=0) #select the position where the gradient is the lowest
    Lx = np.choose(Jmin, Lx)
    Ly = np.choose(Jmin, Ly)

    #Lx = np.mean(Lx,axis=0)
    #Ly = np.mean(Ly,axis=0)

    # Finding a Local Extrema at Mid-points
    # 
    # Mid-points coordinates for each element
    xmj = 0.5*(xjk[[0,1,2],:] + xjk[[1,2,0],:])
    ymj = 0.5*(yjk[[0,1,2],:] + yjk[[1,2,0],:])


    U12 = Zj + Lx*(xmj[0,:] - xj) + Ly*(ymj[0,:] - yj)
    U23 = Zj + Lx*(xmj[1,:] - xj) + Ly*(ymj[1,:] - yj)
    U31 = Zj + Lx*(xmj[2,:] - xj) + Ly*(ymj[2,:] - yj)


    J12 = (U12 - np.amax(np.array([Zj,Zijk[0,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[0,:]]) ,axis=0 ) - U12 > tol)
    J23 = (U23 - np.amax(np.array([Zj,Zijk[1,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[1,:]]) ,axis=0 ) - U23 > tol)
    J31 = (U31 - np.amax(np.array([Zj,Zijk[2,:]]), axis=0) > tol) | (np.amin(np.array([Zj,Zijk[2,:]]) ,axis=0 ) - U31 > tol)

    J = (J12 | J23 | J31)

    #print(np.count_nonzero(~J))

    Lx[J] = 0.0
    Ly[J] = 0.0

    return Lx, Ly

def set_constant_midpoint_values(Xj):
    """
    This function extends the current value in array Xj to the midpoints of the triangle

    Parameters
    ----------
        Xj  : Cell-Centered values of quantity X, [1, nElem]

    Returns
    -------
        Xmj : Mid-points interpolation of quatity Xj, [3, nElem]
    """
    #Reconstruction at mid-point coordinates
    Xmj = np.array([Xj,Xj,Xj])

    return Xmj

def set_linear_midpoint_values(Xj, Lx, Ly, xj, yj, xjk, yjk):
    """
    This function extends the current value in array Xj to the midpoints of the triangle

    Parameters
    ----------
        Xj  : Cell-Centered values of quantity X, [1, nElem]
        Lx  : Partial derivative of X along x-direction, [1, nElem]
        Ly  : Partial derivative of X along y-direction, [1, nElem]
        xj  : Coordinate in x-direction of X, [1, nElem]
        yj  : Coordinate in y-direction of X, [1, nElem]

    Returns
    -------
        Xmj : Mid-points interpolation of quatity Xj, [3, nElem]
    """

    #MidPoints Values
    xmjk = 0.5*(xjk[[0,1,2],:] + xjk[[1,2,0], :]) 
    ymjk = 0.5*(yjk[[0,1,2],:] + yjk[[1,2,0], :]) 

    #Interpolation at Mid-Edges: 12, 23, 31
    Xj12 = Xj + Lx*(xmjk[0,:] - xj) + Ly*(ymjk[0,:] - yj)
    Xj23 = Xj + Lx*(xmjk[1,:] - xj) + Ly*(ymjk[1,:] - yj)
    Xj31 = Xj + Lx*(xmjk[2,:] - xj) + Ly*(ymjk[2,:] - yj)

    #Reconstruction at mid-point coordinates
    Xmj = np.array([Xj12, Xj23, Xj31])

    return Xmj

def set_quadratic_midpoint_values(Xj, Lx, Ly, Lxx, Lyy, Lxy, xj, yj, Ix, Iy, Ixy, T, xjk, yjk):

    c = 0.5+np.sqrt(3)/6

    xmjk1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    xmjk2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]

    ymjk1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    ymjk2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]

    Xj121 = Xj + Lx*(xmjk1[0,:]-xj) + Ly*(ymjk1[0,:]-yj) + 0.5*Lxx*((xmjk1[0,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjk1[0,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjk1[0,:]-xj)*(ymjk1[0,:]-yj)-(Ixy/T-xj*yj))
    Xj122 = Xj + Lx*(xmjk2[0,:]-xj) + Ly*(ymjk2[0,:]-yj) + 0.5*Lxx*((xmjk2[0,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjk2[0,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjk2[0,:]-xj)*(ymjk2[0,:]-yj)-(Ixy/T-xj*yj))

    Xj231 = Xj + Lx*(xmjk1[1,:]-xj) + Ly*(ymjk1[1,:]-yj) + 0.5*Lxx*((xmjk1[1,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjk1[1,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjk1[1,:]-xj)*(ymjk1[1,:]-yj)-(Ixy/T-xj*yj))
    Xj232 = Xj + Lx*(xmjk1[1,:]-xj) + Ly*(ymjk2[1,:]-yj) + 0.5*Lxx*((xmjk2[1,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjk2[1,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjk2[1,:]-xj)*(ymjk2[1,:]-yj)-(Ixy/T-xj*yj))
    
    Xj311 = Xj + Lx*(xmjk1[2,:]-xj) + Ly*(ymjk1[2,:]-yj) + 0.5*Lxx*((xmjk1[2,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjk1[2,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjk1[2,:]-xj)*(ymjk1[2,:]-yj)-(Ixy/T-xj*yj))
    Xj312 = Xj + Lx*(xmjk1[2,:]-xj) + Ly*(ymjk2[2,:]-yj) + 0.5*Lxx*((xmjk2[2,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((ymjk2[2,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xmjk2[2,:]-xj)*(ymjk2[2,:]-yj)-(Ixy/T-xj*yj))

    Xmj1 = np.array([Xj121, Xj231, Xj311])
    Xmj2 = np.array([Xj122, Xj232, Xj312])

    return Xmj1,Xmj2

def set_weno_midpoint_values(Xj,recon,xjk,yjk,ov):

    c = 0.5+np.sqrt(3)/6

    xmjk1 = c*xjk[[0,1,2],:]+(1-c)*xjk[[1,2,0],:]
    xmjk2 = c*xjk[[1,2,0],:]+(1-c)*xjk[[0,1,2],:]

    ymjk1 = c*yjk[[0,1,2],:]+(1-c)*yjk[[1,2,0],:]
    ymjk2 = c*yjk[[1,2,0],:]+(1-c)*yjk[[0,1,2],:]

    Xj121 = recon(xmjk1[0,:],ymjk1[0,:],ov)
    Xj231 = recon(xmjk1[1,:],ymjk1[1,:],ov)
    Xj311 = recon(xmjk1[2,:],ymjk1[2,:],ov)

    Xj122 = recon(xmjk2[0,:],ymjk2[0,:],ov)
    Xj232 = recon(xmjk2[1,:],ymjk2[1,:],ov)
    Xj312 = recon(xmjk2[2,:],ymjk2[2,:],ov)

    Xmj1 = np.array([Xj121, Xj231, Xj311])
    Xmj2 = np.array([Xj122, Xj232, Xj312])

    return Xmj1,Xmj2

def set_constant_vertices_values(Xj):

    return np.array([Xj,Xj,Xj])

def set_linear_vertices_values(Xj, Lx, Ly, xj, yj, xjk, yjk):
    """
    This function extends the current value in array Xj to the vertices of the triangle

    Parameters
    ----------
        Xj  : Cell-Centered values of quantity X, [1, nElem]
        Lx  : Partial derivative of X along x-direction, [1, nElem]
        Ly  : Partial derivative of X along y-direction, [1, nElem]
        xj  : Coordinate in x-direction of X, [1, nElem]
        yj  : Coordinate in y-direction of X, [1, nElem]
        xjk : Coordinate in x-direction of the triangle vertices for each element [3, nElem]
        yjk : Coordinate in y-direction of the triangle vertices for each element [3, nElem]

    Returns
    -------
        Xjk : Mid-points interpolation of quatity Xj, [3, nElem]
    """



    #Values at vertices: 12, 23, 31
    Xj12 = Xj + Lx*(xjk[0,:] - xj) + Ly*(yjk[0,:] - yj)
    Xj23 = Xj + Lx*(xjk[1,:] - xj) + Ly*(yjk[1,:] - yj)
    Xj31 = Xj + Lx*(xjk[2,:] - xj) + Ly*(yjk[2,:] - yj)

    #Reconstruction at vertices coordinates
    Xjk = np.array([Xj12, Xj23, Xj31])

    return Xjk

def set_quadratic_vertices_values(Xj, Lx, Ly, Lxx, Lyy, Lxy, xj, yj, Ix, Iy, Ixy, T, xjk1, yjk1):

    Xj12 = Xj + Lx*(xjk1[0,:]-xj) + Ly*(yjk1[0,:]-yj) + 0.5*Lxx*((xjk1[0,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((yjk1[0,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xjk1[0,:]-xj)*(yjk1[0,:]-yj)-(Ixy/T-xj*yj))

    Xj23 = Xj + Lx*(xjk1[1,:]-xj) + Ly*(yjk1[1,:]-yj) + 0.5*Lxx*((xjk1[1,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((yjk1[1,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xjk1[1,:]-xj)*(yjk1[1,:]-yj)-(Ixy/T-xj*yj))
    
    Xj31 = Xj + Lx*(xjk1[2,:]-xj) + Ly*(yjk1[2,:]-yj) + 0.5*Lxx*((xjk1[2,:]-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((yjk1[2,:]-yj)**2-(Ix/T-yj**2)) + Lxy*((xjk1[2,:]-xj)*(yjk1[2,:]-yj)-(Ixy/T-xj*yj))

    Xjk = np.array([Xj12, Xj23, Xj31])

    return Xjk

def set_weno_vertices_values(Xj,recon,xjk,yjk,ov):
    Xj12 = recon(xjk[0,:],yjk[0,:],ov)
    Xj23 = recon(xjk[1,:],yjk[1,:],ov)
    Xj31 = recon(xjk[2,:],yjk[2,:],ov)

    return np.array([Xj12, Xj23, Xj31])

def set_linear_vertices_values_special(Xj, Xjk, Bjk, Lx, Ly, xj, yj, xjk, yjk,TOL):
    """
    This function extends the water surface to the vertices of the triangle on the wet cells and snaps the water surface to the bathymetry on the dry cells

    Parameters
    ----------
        Xj  : Cell-Centered values of quantity X, [1, nElem]
        Lx  : Partial derivative of X along x-direction, [1, nElem]
        Ly  : Partial derivative of X along y-direction, [1, nElem]
        xj  : Coordinate in x-direction of X, [1, nElem]
        yj  : Coordinate in y-direction of X, [1, nElem]
        xjk : Coordinate in x-direction of the triangle vertices for each element [3, nElem]
        yjk : Coordinate in y-direction of the triangle vertices for each element [3, nElem]

    Returns
    -------
        Xjk : Mid-points interpolation of quatity Xj, [3, nElem]
    """



    #Values at vertices: 12, 23, 31
    Xj12 = Xj + Lx*(xjk[0,:] - xj) + Ly*(yjk[0,:] - yj)
    Xj23 = Xj + Lx*(xjk[1,:] - xj) + Ly*(yjk[1,:] - yj)
    Xj31 = Xj + Lx*(xjk[2,:] - xj) + Ly*(yjk[2,:] - yj)

    wet = np.any(Xjk-TOL>Bjk, axis=0)
    dry = ~wet

    Xjk=Xjk.T

    #Reconstruction at vertices coordinates
    Xjk[wet] = np.array([Xj12[wet], Xj23[wet], Xj31[wet]]).T
    Xjk[dry] = np.array([Bjk.T[dry][:,0], Bjk.T[dry][:,1], Bjk.T[dry][:,2]]).T

    Xjk=Xjk.T

    return Xjk

def set_quadratic_in_values(Xj, Lx, Ly, Lxx, Lyy, Lxy, xj, yj, Ix, Iy, Ixy, T, xg0, xg1, xg2, yg0, yg1, yg2):

    Xg0 = Xj + Lx*(xg0-xj) + Ly*(yg0-yj) + 0.5*Lxx*((xg0-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((yg0-yj)**2-(Ix/T-yj**2)) + Lxy*((xg0-xj)*(yg0-yj)-(Ixy/T-xj*yj))

    Xg1 = Xj + Lx*(xg1-xj) + Ly*(yg1-yj) + 0.5*Lxx*((xg1-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((yg1-yj)**2-(Ix/T-yj**2)) + Lxy*((xg1-xj)*(yg1-yj)-(Ixy/T-xj*yj))
    
    Xg2 = Xj + Lx*(xg2-xj) + Ly*(yg2-yj) + 0.5*Lxx*((xg2-xj)**2-(Iy/T-xj**2)) + 0.5*Lyy*((yg2-yj)**2-(Ix/T-yj**2)) + Lxy*((xg2-xj)*(yg2-yj)-(Ixy/T-xj*yj))

    return Xg0,Xg1,Xg2

def set_weno_in_values(recon,xg0,xg1,xg2,yg0,yg1,yg2,ov):
    return recon(xg0,yg0,ov), recon(xg1,yg1,ov), recon(xg2,yg2,ov)

def set_quadratic_din_values(Lx,Ly,Lxx,Lyy,Lxy,xj,yj,Ix,Iy,Ixy,T,xg0,xg1,xg2,yg0,yg1,yg2):
    wx1 = Lx+Lxx*(xg0-xj)+Lxy*(yg0-yj)
    wx2 = Lx+Lxx*(xg1-xj)+Lxy*(yg1-yj)
    wx3 = Lx+Lxx*(xg2-xj)+Lxy*(yg2-yj)

    wy1 = Ly+Lyy*(yg0-yj)+Lxy*(xg0-xj)
    wy2 = Ly+Lyy*(yg1-yj)+Lxy*(xg1-xj)
    wy3 = Ly+Lyy*(yg2-yj)+Lxy*(xg2-xj)

    return wx1,wx2,wx3,wy1,wy2,wy3

def set_constant_well_balanced_wet_dry(Wjkr, Bjkr, Wmjr, Hmjr, Bmjr, Bjr, Wjr, xjr, yjr, xjkr, yjkr, TOL, DRY, jk, Ghosts):
    """
    This function reconstructs the Water Level after the water depth Hmj has been positived

    Parameters
    ----------
        Hmj : Mid-point water depth, [3, nElem]
        Wmj : Mid-points values of water depth, [3, nElem]
        Bmj : Mid-points values of bathymetry, [3, nElem]
        DRY : Tolerance that defines when a cell is dry, [1,1]
        jk  : Global indeces for side neighbor cells quantities, [1, 3*nElem]
    
    Returns
    -------
        Hmj : Corrected values of water depth at mid-point, [3, nElem]
        Wmj : Corrected values of water level at mid-point, [3, nElem]
    """
    #Left and Right Quantities
    Waux = Wmjr.take(jk)
    Zaux = np.maximum(Bmjr - Wmjr, Bmjr - Waux)
    Zaux = np.maximum(0.000, Zaux)

    #Well-balanced reconstruction at dry cells
    Wfake = Bmjr + Hmjr
    Wfake = Wfake - Zaux
    
    Baux = Bmjr - Zaux
    Hfake = np.maximum(0.01*DRY, Wfake - Baux)

    return Wjkr, Wfake, Hfake

def set_linear_well_balanced_wet_dry(Wjkr, Bjkr, Wmjr, Hmjr, Bmjr, Bjr, Wjr, xjr, yjr, xjkr, yjkr, TOL, DRY, jk, Ghosts,wxr,wyr):
    """
    This function reconstructs the Water Level after the water depth Hmj has been positived

    Parameters
    ----------
        Wjk : Vertices water level, [3, nElem]
        Bjk : Vertices values of Bathymetry, [3, nElem]
        xjk : Coordinate in x-direction of the triangle vertices for each element [3, nElem]
        yjk : Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    Returns
    -------
        Hjk : Corrected values of water depth at vertices,  [3, nElem]
        Wjk : Corrected values of water level at vertices,  [3, nElem]
        Hmj : Corrected values of water depth at midpionts, [3, nElem]
    """
    Wpos=Wjkr[:,:Ghosts[0,0]]   #Cutting ghost cells from analysis
    Bjk=Bjkr[:,:Ghosts[0,0]]    #Cutting ghost cells from analysis

    aux=Wpos-Bjk #Vector with near 0s where we have to reconstruct
    Wpos=np.where(aux<TOL,Bjk,Wpos) #cutoff values
    aux=np.where(aux<TOL,0,aux)     #cutoff values
    Wjr[Wjr<Bjr]=Bjr[Wjr<Bjr]       #cutoff values

    one_dry_point  = np.where(np.count_nonzero(aux,axis=0)==2)[0]   #Finds elements where there's one dry point (because there's two nonzero)
    two_dry_point = np.where(np.count_nonzero(aux,axis=0)==1)[0]    #Finds elements where there's two dry points (because there's one nonzero)
    three_dry_point = np.where(np.count_nonzero(aux,axis=0)==0)[0]  #Finds elements where there's three dry points (because there's no nonzero)

    block1=Wpos[:,one_dry_point].T      #Builds block to correct of elements with one dry point
    block2=Wpos[:,two_dry_point].T      #Builds block to correct of elements with two dry points
    block3=Wpos[:,three_dry_point].T    #Builds block to correct of elements with three dry points

    auxblock1=aux[:,one_dry_point].T    #Builds auxiliary block with 0s where there's dry points
    auxblock2=aux[:,two_dry_point].T    #Builds auxiliary block with 0s where there's dry points
    auxblock3=aux[:,three_dry_point].T  #Builds auxiliary block with 0s where there's dry points

    xjkm = 0.5*(xjkr[[0,1,2],:] + xjkr[[1,2,0], :])     #Finds x coordinates of the midpoints
    yjkm = 0.5*(yjkr[[0,1,2],:] + yjkr[[1,2,0], :])     #Finds y coordinates of the midpoints

    where_dry_1=np.where(auxblock1==0)      #Finds indices of dry vertex in the one-dry-point block by finding 0s in the aux block
    where_notdry_1=np.where(auxblock1)      #Finds indices of wet vertices in the one-dry-point block by finding non 0s in the aux block
    where_dry_2=np.where(auxblock2==0)      #Finds indices of dry vertices in the two-dry-point block by finding 0s in the aux block
    where_notdry_2=np.where(auxblock2)      #Finds indices of wet vertex in the two-dry-point block by finding non 0s in the aux block
    where_dry_3=np.where(auxblock3==0)      #Finds indices of dry vertices in the three-dry-point-block by finding 0s in the aux block

    #Handling elements with one dry vertex

    enum1=np.array(range(len(block1))).astype(int)  #Builds an array with the enumeration of elements in block1

    where_notdry_1_vertex1=where_notdry_1[1][1::2]  #Gives the indices of the 1st wet vertex of each element
    where_notdry_1_vertex2=where_notdry_1[1][0::2]  #Gives the indices of the 2nd wet vertex of each element

    Bjk1_one=Bjk.T[one_dry_point,where_dry_1[1]]          #Gives the bottom height at the dry vertex (xjk1,yjk1)
    Bjk2_one=Bjk.T[one_dry_point,where_notdry_1_vertex1]  #Gives the bottom height at the 1st wet vertex (xjk2,yjk2)

    xjk1_one=xjkr.T[one_dry_point,where_dry_1[1]]           #Gives the x coordinate of the dry vertex
    xjk2_one=xjkr.T[one_dry_point,where_notdry_1_vertex1]   #Gives the x coordinate of the 1st wet vertex
    xjk3_one=xjkr.T[one_dry_point,where_notdry_1_vertex2]   #Gives the x coordinate of the 2nd wet vertex

    yjk1_one=yjkr.T[one_dry_point,where_dry_1[1]]           #Gives the y coordinate of the dry vertex
    yjk2_one=yjkr.T[one_dry_point,where_notdry_1_vertex1]   #Gives the y coordinate of the 1st wet vertex
    yjk3_one=yjkr.T[one_dry_point,where_notdry_1_vertex2]   #Gives the y coordinate of the 2nd wet vertex

    xj_one=xjr.T[one_dry_point]     #Gives the x coordinate of the element's center
    yj_one=yjr.T[one_dry_point]     #Gives the y coordinate of the element's center
    wj_one=Wjr.T[one_dry_point]     #Gives the water height at the element's center (average over the cell)
    Bj_one=Bjr.T[one_dry_point]     #Gives the bottom height at the element's center

    W = (3/2)*(wj_one-Bj_one)+Bjk2_one  #  Calculates the water height at the 1st wet vertex
    block1[(enum1,where_notdry_1_vertex1)]=W        #Replaces water height at 1st wet vertex in block

    new_wx1 = 1.*( Bjk1_one*yj_one - W*yj_one + W*yjk1_one - wj_one*yjk1_one - Bjk1_one*yjk2_one + wj_one*yjk2_one)/(xjk1_one*yj_one - xjk2_one*yj_one - xj_one*yjk1_one + xjk2_one*yjk1_one + xj_one*yjk2_one - xjk1_one*yjk2_one)
    new_wy1 = 1.*(-Bjk1_one*xj_one + W*xj_one - W*xjk1_one + wj_one*xjk1_one + Bjk1_one*xjk2_one - wj_one*xjk2_one)/(xjk1_one*yj_one - xjk2_one*yj_one - xj_one*yjk1_one + xjk2_one*yjk1_one + xj_one*yjk2_one - xjk1_one*yjk2_one)

    wjk3 = wj_one+new_wx1*(xjk3_one-xj_one)+new_wy1*(yjk3_one-yj_one) #Calculates water surface at 2nd wet bertex
    block1[(enum1,where_notdry_1_vertex2)]=wjk3     #Replaces water height at 2nd wet vertex in block

    Wmjr[0,one_dry_point]=wj_one+new_wx1*(xjkm[0,one_dry_point]-xj_one)+new_wy1*(yjkm[0,one_dry_point]-yj_one)
    Wmjr[1,one_dry_point]=wj_one+new_wx1*(xjkm[1,one_dry_point]-xj_one)+new_wy1*(yjkm[1,one_dry_point]-yj_one)
    Wmjr[2,one_dry_point]=wj_one+new_wx1*(xjkm[2,one_dry_point]-xj_one)+new_wy1*(yjkm[2,one_dry_point]-yj_one)

    #Handling elements with two dry vertices
    #Analogous to one dry vertex

    enum2=np.array(range(len(block2))).astype(int)

    where_dry_2_vertex1=where_dry_2[1][1::2]
    where_dry_2_vertex2=where_dry_2[1][0::2]

    Bjk1_two=Bjk.T[two_dry_point,where_dry_2_vertex1]
    Bjk2_two=Bjk.T[two_dry_point,where_dry_2_vertex2]

    xjk1_two=xjkr.T[two_dry_point,where_dry_2_vertex1]
    xjk2_two=xjkr.T[two_dry_point,where_dry_2_vertex2]
    xjk3_two=xjkr.T[two_dry_point,where_notdry_2[1]]

    yjk1_two=yjkr.T[two_dry_point,where_dry_2_vertex1]
    yjk2_two=yjkr.T[two_dry_point,where_dry_2_vertex2]
    yjk3_two=yjkr.T[two_dry_point,where_notdry_2[1]]

    xj_two=xjr.T[two_dry_point]
    yj_two=yjr.T[two_dry_point]
    wj_two=Wjr.T[two_dry_point]
    Bj_two=Bjr.T[two_dry_point]

    W = Bjk2_two

    new_wx2 = 1.*( Bjk1_two*yj_two - W*yj_two + W*yjk1_two - wj_two*yjk1_two - Bjk1_two*yjk2_two + wj_two*yjk2_two)/(xjk1_two*yj_two - xjk2_two*yj_two - xj_two*yjk1_two + xjk2_two*yjk1_two + xj_two*yjk2_two - xjk1_two*yjk2_two)
    new_wy2 = 1.*(-Bjk1_two*xj_two + W*xj_two - W*xjk1_two + wj_two*xjk1_two + Bjk1_two*xjk2_two - wj_two*xjk2_two)/(xjk1_two*yj_two - xjk2_two*yj_two - xj_two*yjk1_two + xjk2_two*yjk1_two + xj_two*yjk2_two - xjk1_two*yjk2_two)

    Wmjr[0,two_dry_point]=wj_two+new_wx2*(xjkm[0,two_dry_point]-xj_two)+new_wy2*(yjkm[0,two_dry_point]-yj_two)
    Wmjr[1,two_dry_point]=wj_two+new_wx2*(xjkm[1,two_dry_point]-xj_two)+new_wy2*(yjkm[1,two_dry_point]-yj_two)
    Wmjr[2,two_dry_point]=wj_two+new_wx2*(xjkm[2,two_dry_point]-xj_two)+new_wy2*(yjkm[2,two_dry_point]-yj_two)

    wjk3=wj_two+new_wx2*(xjk3_two-xj_two)+new_wy2*(yjk3_two-yj_two)

    block2[(enum2,where_notdry_2[1])]=wjk3

    #Handling elements with three dry vertices

    block3[where_dry_3]=Bjk[:,three_dry_point].T[where_dry_3]    #Makes water height equal to bottom height where it should be dry

    enum3=np.array(range(len(block3))).astype(int)

    where_dry_3_vertex1=where_dry_3[1][0::3]
    where_dry_3_vertex2=where_dry_3[1][1::3]
    where_dry_3_vertex3=where_dry_3[1][2::3]

    Bjk1_three=Bjk.T[three_dry_point,where_dry_3_vertex1]
    Bjk2_three=Bjk.T[three_dry_point,where_dry_3_vertex2]
    Bjk3_three=Bjk.T[three_dry_point,where_dry_3_vertex3]

    xjk1_three=xjkr.T[three_dry_point,where_dry_3_vertex1]
    xjk2_three=xjkr.T[three_dry_point,where_dry_3_vertex2]
    xjk3_three=xjkr.T[three_dry_point,where_dry_3_vertex3]

    yjk1_three=yjkr.T[three_dry_point,where_dry_3_vertex1]
    yjk2_three=yjkr.T[three_dry_point,where_dry_3_vertex2]
    yjk3_three=yjkr.T[three_dry_point,where_dry_3_vertex3]

    xj_three=xjr.T[three_dry_point]
    yj_three=yjr.T[three_dry_point]
    wj_three=Wjr.T[three_dry_point]
    Bj_three=Bjr.T[three_dry_point]

    W = Bjk2_three

    new_wx3 = 1.*( Bjk1_three*yj_three - W*yj_three + W*yjk1_three - wj_three*yjk1_three - Bjk1_three*yjk2_three + wj_three*yjk2_three)/(xjk1_three*yj_three - xjk2_three*yj_three - xj_three*yjk1_three + xjk2_three*yjk1_three + xj_three*yjk2_three - xjk1_three*yjk2_three)
    new_wy3 = 1.*(-Bjk1_three*xj_three + W*xj_three - W*xjk1_three + wj_three*xjk1_three + Bjk1_three*xjk2_three - wj_three*xjk2_three)/(xjk1_three*yj_three - xjk2_three*yj_three - xj_three*yjk1_three + xjk2_three*yjk1_three + xj_three*yjk2_three - xjk1_three*yjk2_three)

    Wmjr[0,three_dry_point]=wj_three+new_wx3*(xjkm[0,three_dry_point]-xj_three)+new_wy3*(yjkm[0,three_dry_point]-yj_three)
    Wmjr[1,three_dry_point]=wj_three+new_wx3*(xjkm[1,three_dry_point]-xj_three)+new_wy3*(yjkm[1,three_dry_point]-yj_three)
    Wmjr[2,three_dry_point]=wj_three+new_wx3*(xjkm[2,three_dry_point]-xj_three)+new_wy3*(yjkm[2,three_dry_point]-yj_three)

    #Correcting the grid

    Wjkr[:,one_dry_point]=block1.T
    Wjkr[:,two_dry_point]=block2.T
    Wjkr[:,three_dry_point]=block3.T

    Hmjr = np.maximum(0.001*DRY, Wmjr - Bmjr) #Calculating water column at midpoints

    wxr[one_dry_point]=new_wx1
    wyr[one_dry_point]=new_wy1
    wxr[two_dry_point]=new_wx2
    wyr[two_dry_point]=new_wx2
    wxr[three_dry_point]=new_wx3
    wyr[three_dry_point]=new_wx3

    return Wjkr, Wmjr, Hmjr, wxr, wyr

def set_linear_well_balanced_wet_dry2(Wjkr, Bjkr, Wmjr1, Wmjr2, Hmjr1, Hmjr2, Bmjr1, Bmjr2, Bjr, Wjr, xjr, yjr, xjkr, yjkr, TOL, DRY, jk, Ghosts,wxr,wyr,reconr):
    """
    This function reconstructs the Water Level after the water depth Hmj has been positived

    Parameters
    ----------
        Wjk : Vertices water level, [3, nElem]
        Bjk : Vertices values of Bathymetry, [3, nElem]
        xjk : Coordinate in x-direction of the triangle vertices for each element [3, nElem]
        yjk : Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    Returns
    -------
        Hjk : Corrected values of water depth at vertices,  [3, nElem]
        Wjk : Corrected values of water level at vertices,  [3, nElem]
        Hmj : Corrected values of water depth at midpionts, [3, nElem]
    """
    Wpos=Wjkr[:,:Ghosts[0,0]]   #Cutting ghost cells from analysis
    Bjk=Bjkr[:,:Ghosts[0,0]]    #Cutting ghost cells from analysis

    aux=Wpos-Bjk #Vector with near 0s where we have to reconstruct
    Wpos=np.where(aux<TOL,Bjk,Wpos) #cutoff values
    aux=np.where(aux<TOL,0,aux)     #cutoff values
    Wjr[Wjr<Bjr]=Bjr[Wjr<Bjr]       #cutoff values

    one_dry_point  = np.where(np.count_nonzero(aux,axis=0)==2)[0]   #Finds elements where there's one dry point (because there's two nonzero)
    two_dry_point = np.where(np.count_nonzero(aux,axis=0)==1)[0]    #Finds elements where there's two dry points (because there's one nonzero)
    three_dry_point = np.where(np.count_nonzero(aux,axis=0)==0)[0]  #Finds elements where there's three dry points (because there's no nonzero)

    block1=Wpos[:,one_dry_point].T      #Builds block to correct of elements with one dry point
    block2=Wpos[:,two_dry_point].T      #Builds block to correct of elements with two dry points
    block3=Wpos[:,three_dry_point].T    #Builds block to correct of elements with three dry points

    auxblock1=aux[:,one_dry_point].T    #Builds auxiliary block with 0s where there's dry points
    auxblock2=aux[:,two_dry_point].T    #Builds auxiliary block with 0s where there's dry points
    auxblock3=aux[:,three_dry_point].T  #Builds auxiliary block with 0s where there's dry points

    c = 0.5+np.sqrt(3)/6

    xjkm1 = c*xjkr[[0,1,2],:]+(1-c)*xjkr[[1,2,0],:] #Coordinates of the midpoints
    xjkm2 = c*xjkr[[1,2,0],:]+(1-c)*xjkr[[0,1,2],:]
    yjkm1 = c*yjkr[[0,1,2],:]+(1-c)*yjkr[[1,2,0],:]
    yjkm2 = c*yjkr[[1,2,0],:]+(1-c)*yjkr[[0,1,2],:]

    where_dry_1=np.where(auxblock1==0)      #Finds indices of dry vertex in the one-dry-point block by finding 0s in the aux block
    where_notdry_1=np.where(auxblock1)      #Finds indices of wet vertices in the one-dry-point block by finding non 0s in the aux block
    where_dry_2=np.where(auxblock2==0)      #Finds indices of dry vertices in the two-dry-point block by finding 0s in the aux block
    where_notdry_2=np.where(auxblock2)      #Finds indices of wet vertex in the two-dry-point block by finding non 0s in the aux block
    where_dry_3=np.where(auxblock3==0)      #Finds indices of dry vertices in the three-dry-point-block by finding 0s in the aux block

    #Handling elements with one dry vertex

    enum1=np.array(range(len(block1))).astype(int)  #Builds an array with the enumeration of elements in block1

    where_notdry_1_vertex1=where_notdry_1[1][1::2]  #Gives the indices of the 1st wet vertex of each element
    where_notdry_1_vertex2=where_notdry_1[1][0::2]  #Gives the indices of the 2nd wet vertex of each element

    Bjk1_one=Bjk.T[one_dry_point,where_dry_1[1]]          #Gives the bottom height at the dry vertex (xjk1,yjk1)
    Bjk2_one=Bjk.T[one_dry_point,where_notdry_1_vertex1]  #Gives the bottom height at the 1st wet vertex (xjk2,yjk2)

    xjk1_one=xjkr.T[one_dry_point,where_dry_1[1]]           #Gives the x coordinate of the dry vertex
    xjk2_one=xjkr.T[one_dry_point,where_notdry_1_vertex1]   #Gives the x coordinate of the 1st wet vertex
    xjk3_one=xjkr.T[one_dry_point,where_notdry_1_vertex2]   #Gives the x coordinate of the 2nd wet vertex

    yjk1_one=yjkr.T[one_dry_point,where_dry_1[1]]           #Gives the y coordinate of the dry vertex
    yjk2_one=yjkr.T[one_dry_point,where_notdry_1_vertex1]   #Gives the y coordinate of the 1st wet vertex
    yjk3_one=yjkr.T[one_dry_point,where_notdry_1_vertex2]   #Gives the y coordinate of the 2nd wet vertex

    xj_one=xjr.T[one_dry_point]     #Gives the x coordinate of the element's center
    yj_one=yjr.T[one_dry_point]     #Gives the y coordinate of the element's center
    wj_one=Wjr.T[one_dry_point]     #Gives the water height at the element's center (average over the cell)
    Bj_one=Bjr.T[one_dry_point]     #Gives the bottom height at the element's center

    W = (3/2)*(wj_one-Bj_one)+Bjk2_one  #  Calculates the water height at the 1st wet vertex
    block1[(enum1,where_notdry_1_vertex1)]=W        #Replaces water height at 1st wet vertex in block

    new_wx1 = 1.*( Bjk1_one*yj_one - W*yj_one + W*yjk1_one - wj_one*yjk1_one - Bjk1_one*yjk2_one + wj_one*yjk2_one)/(xjk1_one*yj_one - xjk2_one*yj_one - xj_one*yjk1_one + xjk2_one*yjk1_one + xj_one*yjk2_one - xjk1_one*yjk2_one)
    new_wy1 = 1.*(-Bjk1_one*xj_one + W*xj_one - W*xjk1_one + wj_one*xjk1_one + Bjk1_one*xjk2_one - wj_one*xjk2_one)/(xjk1_one*yj_one - xjk2_one*yj_one - xj_one*yjk1_one + xjk2_one*yjk1_one + xj_one*yjk2_one - xjk1_one*yjk2_one)

    wjk3 = wj_one+new_wx1*(xjk3_one-xj_one)+new_wy1*(yjk3_one-yj_one) #Calculates water surface at 2nd wet bertex
    block1[(enum1,where_notdry_1_vertex2)]=wjk3     #Replaces water height at 2nd wet vertex in block

    Wmjr1[0,one_dry_point]=wj_one+new_wx1*(xjkm1[0,one_dry_point]-xj_one)+new_wy1*(yjkm1[0,one_dry_point]-yj_one)
    Wmjr1[1,one_dry_point]=wj_one+new_wx1*(xjkm1[1,one_dry_point]-xj_one)+new_wy1*(yjkm1[1,one_dry_point]-yj_one)
    Wmjr1[2,one_dry_point]=wj_one+new_wx1*(xjkm1[2,one_dry_point]-xj_one)+new_wy1*(yjkm1[2,one_dry_point]-yj_one)

    Wmjr2[0,one_dry_point]=wj_one+new_wx1*(xjkm2[0,one_dry_point]-xj_one)+new_wy1*(yjkm2[0,one_dry_point]-yj_one)
    Wmjr2[1,one_dry_point]=wj_one+new_wx1*(xjkm2[1,one_dry_point]-xj_one)+new_wy1*(yjkm2[1,one_dry_point]-yj_one)
    Wmjr2[2,one_dry_point]=wj_one+new_wx1*(xjkm2[2,one_dry_point]-xj_one)+new_wy1*(yjkm2[2,one_dry_point]-yj_one)

    #Handling elements with two dry vertices
    #Analogous to one dry vertex

    enum2=np.array(range(len(block2))).astype(int)

    where_dry_2_vertex1=where_dry_2[1][1::2]
    where_dry_2_vertex2=where_dry_2[1][0::2]

    Bjk1_two=Bjk.T[two_dry_point,where_dry_2_vertex1]
    Bjk2_two=Bjk.T[two_dry_point,where_dry_2_vertex2]

    xjk1_two=xjkr.T[two_dry_point,where_dry_2_vertex1]
    xjk2_two=xjkr.T[two_dry_point,where_dry_2_vertex2]
    xjk3_two=xjkr.T[two_dry_point,where_notdry_2[1]]

    yjk1_two=yjkr.T[two_dry_point,where_dry_2_vertex1]
    yjk2_two=yjkr.T[two_dry_point,where_dry_2_vertex2]
    yjk3_two=yjkr.T[two_dry_point,where_notdry_2[1]]

    xj_two=xjr.T[two_dry_point]
    yj_two=yjr.T[two_dry_point]
    wj_two=Wjr.T[two_dry_point]
    Bj_two=Bjr.T[two_dry_point]

    W = Bjk2_two

    new_wx2 = 1.*( Bjk1_two*yj_two - W*yj_two + W*yjk1_two - wj_two*yjk1_two - Bjk1_two*yjk2_two + wj_two*yjk2_two)/(xjk1_two*yj_two - xjk2_two*yj_two - xj_two*yjk1_two + xjk2_two*yjk1_two + xj_two*yjk2_two - xjk1_two*yjk2_two)
    new_wy2 = 1.*(-Bjk1_two*xj_two + W*xj_two - W*xjk1_two + wj_two*xjk1_two + Bjk1_two*xjk2_two - wj_two*xjk2_two)/(xjk1_two*yj_two - xjk2_two*yj_two - xj_two*yjk1_two + xjk2_two*yjk1_two + xj_two*yjk2_two - xjk1_two*yjk2_two)

    Wmjr1[0,two_dry_point]=wj_two+new_wx2*(xjkm1[0,two_dry_point]-xj_two)+new_wy2*(yjkm1[0,two_dry_point]-yj_two)
    Wmjr1[1,two_dry_point]=wj_two+new_wx2*(xjkm1[1,two_dry_point]-xj_two)+new_wy2*(yjkm1[1,two_dry_point]-yj_two)
    Wmjr1[2,two_dry_point]=wj_two+new_wx2*(xjkm1[2,two_dry_point]-xj_two)+new_wy2*(yjkm1[2,two_dry_point]-yj_two)

    Wmjr2[0,two_dry_point]=wj_two+new_wx2*(xjkm2[0,two_dry_point]-xj_two)+new_wy2*(yjkm2[0,two_dry_point]-yj_two)
    Wmjr2[1,two_dry_point]=wj_two+new_wx2*(xjkm2[1,two_dry_point]-xj_two)+new_wy2*(yjkm2[1,two_dry_point]-yj_two)
    Wmjr2[2,two_dry_point]=wj_two+new_wx2*(xjkm2[2,two_dry_point]-xj_two)+new_wy2*(yjkm2[2,two_dry_point]-yj_two)

    wjk3=wj_two+new_wx2*(xjk3_two-xj_two)+new_wy2*(yjk3_two-yj_two)

    block2[(enum2,where_notdry_2[1])]=wjk3

    #Handling elements with three dry vertices

    block3[where_dry_3]=Bjk[:,three_dry_point].T[where_dry_3]    #Makes water height equal to bottom height where it should be dry

    enum3=np.array(range(len(block3))).astype(int)

    where_dry_3_vertex1=where_dry_3[1][0::3]
    where_dry_3_vertex2=where_dry_3[1][1::3]
    where_dry_3_vertex3=where_dry_3[1][2::3]

    Bjk1_three=Bjk.T[three_dry_point,where_dry_3_vertex1]
    Bjk2_three=Bjk.T[three_dry_point,where_dry_3_vertex2]
    Bjk3_three=Bjk.T[three_dry_point,where_dry_3_vertex3]

    xjk1_three=xjkr.T[three_dry_point,where_dry_3_vertex1]
    xjk2_three=xjkr.T[three_dry_point,where_dry_3_vertex2]
    xjk3_three=xjkr.T[three_dry_point,where_dry_3_vertex3]

    yjk1_three=yjkr.T[three_dry_point,where_dry_3_vertex1]
    yjk2_three=yjkr.T[three_dry_point,where_dry_3_vertex2]
    yjk3_three=yjkr.T[three_dry_point,where_dry_3_vertex3]

    xj_three=xjr.T[three_dry_point]
    yj_three=yjr.T[three_dry_point]
    wj_three=Wjr.T[three_dry_point]
    Bj_three=Bjr.T[three_dry_point]

    W = Bjk2_three

    new_wx3 = 1.*( Bjk1_three*yj_three - W*yj_three + W*yjk1_three - wj_three*yjk1_three - Bjk1_three*yjk2_three + wj_three*yjk2_three)/(xjk1_three*yj_three - xjk2_three*yj_three - xj_three*yjk1_three + xjk2_three*yjk1_three + xj_three*yjk2_three - xjk1_three*yjk2_three)
    new_wy3 = 1.*(-Bjk1_three*xj_three + W*xj_three - W*xjk1_three + wj_three*xjk1_three + Bjk1_three*xjk2_three - wj_three*xjk2_three)/(xjk1_three*yj_three - xjk2_three*yj_three - xj_three*yjk1_three + xjk2_three*yjk1_three + xj_three*yjk2_three - xjk1_three*yjk2_three)

    Wmjr1[0,three_dry_point]=wj_three+new_wx3*(xjkm1[0,three_dry_point]-xj_three)+new_wy3*(yjkm1[0,three_dry_point]-yj_three)
    Wmjr1[1,three_dry_point]=wj_three+new_wx3*(xjkm1[1,three_dry_point]-xj_three)+new_wy3*(yjkm1[1,three_dry_point]-yj_three)
    Wmjr1[2,three_dry_point]=wj_three+new_wx3*(xjkm1[2,three_dry_point]-xj_three)+new_wy3*(yjkm1[2,three_dry_point]-yj_three)

    Wmjr2[0,three_dry_point]=wj_three+new_wx3*(xjkm2[0,three_dry_point]-xj_three)+new_wy3*(yjkm2[0,three_dry_point]-yj_three)
    Wmjr2[1,three_dry_point]=wj_three+new_wx3*(xjkm2[1,three_dry_point]-xj_three)+new_wy3*(yjkm2[1,three_dry_point]-yj_three)
    Wmjr2[2,three_dry_point]=wj_three+new_wx3*(xjkm2[2,three_dry_point]-xj_three)+new_wy3*(yjkm2[2,three_dry_point]-yj_three)

    #Correcting the grid

    Wjkr[:,one_dry_point]=block1.T
    Wjkr[:,two_dry_point]=block2.T
    Wjkr[:,three_dry_point]=block3.T

    Hmjr1 = np.maximum(0.01*DRY, Wmjr1 - Bmjr1) #Calculating water column at midpoints
    Hmjr2 = np.maximum(0.01*DRY, Wmjr2 - Bmjr2) #Calculating water column at midpoints

    one=np.zeros_like(xjr)
    one[one_dry_point]=1
    two=np.zeros_like(xjr)
    two[two_dry_point]=1
    three=np.zeros_like(xjr)
    three[three_dry_point]=1

    new_wx3f=np.zeros_like(xjr)
    new_wx3f[three_dry_point]=new_wx3
    new_wx2f=np.zeros_like(xjr)
    new_wx2f[two_dry_point]=new_wx2
    new_wx1f=np.zeros_like(xjr)
    new_wx1f[one_dry_point]=new_wx1

    new_wy3f=np.zeros_like(xjr)
    new_wy3f[three_dry_point]=new_wy3
    new_wy2f=np.zeros_like(xjr)
    new_wy2f[two_dry_point]=new_wy2
    new_wy1f=np.zeros_like(xjr)
    new_wy1f[one_dry_point]=new_wy1

    wj_threef=np.zeros_like(xjr)
    wj_threef[three_dry_point]=wj_three
    wj_twof=np.zeros_like(xjr)
    wj_twof[two_dry_point]=wj_two
    wj_onef=np.zeros_like(xjr)
    wj_onef[one_dry_point]=wj_one

    xj_threef=np.zeros_like(xjr)
    xj_threef[three_dry_point]=xj_three
    xj_twof=np.zeros_like(xjr)
    xj_twof[two_dry_point]=xj_two
    xj_onef=np.zeros_like(xjr)
    xj_onef[one_dry_point]=xj_one
    
    yj_threef=np.zeros_like(xjr)
    yj_threef[three_dry_point]=yj_three
    yj_twof=np.zeros_like(xjr)
    yj_twof[two_dry_point]=yj_two
    yj_onef=np.zeros_like(xjr)
    yj_onef[one_dry_point]=yj_one


    wxrn = lambda x,y,o: wxr(x,y,o)*(1-1.*one)*(1-1.*two)*(1-1.*three)+(new_wx3f)*(three)+(new_wx2f)*(two)+(new_wx1f)*(one)
    wyrn = lambda x,y,o: wyr(x,y,o)*(1-1.*one)*(1-1.*two)*(1-1.*three)+(new_wy3f)*(three)+(new_wy2f)*(two)+(new_wy1f)*(one)

    reconrn = lambda x,y,o: reconr(x,y,o)*(1-1.*one)*(1-1.*two)*(1-1.*three) + (wj_threef+new_wx3f*(x-xj_threef)+new_wy3f*(y-yj_threef))*three + (wj_twof+new_wx2f*(x-xj_twof)+new_wy2f*(y-yj_twof))*two + (wj_onef+new_wx1f*(x-xj_onef)+new_wy1f*(y-yj_onef))*one

    return Wjkr, Wmjr1, Wmjr2, Hmjr1, Hmjr2, wxrn, wyrn, reconrn

def old_set_linear_well_balanced_wet_dry(Wjkr, Bjkr, Wmjr, Hmjr, Bmjr, Bjr, Wjr, xjr, yjr, xjkr, yjkr, TOL, DRY, jk, Ghosts):
    """
    This function reconstructs the Water Level after the water depth Hmj has been positived

    Parameters
    ----------
        Wjk : Vertices water level, [3, nElem]
        Bjk : Vertices values of Bathymetry, [3, nElem]
        xjk : Coordinate in x-direction of the triangle vertices for each element [3, nElem]
        yjk : Coordinate in y-direction of the triangle vertices for each element [3, nElem]
    
    Returns
    -------
        Hjk : Corrected values of water depth at vertices, [3, nElem]
        Wjk : Corrected values of water level at vertices, [3, nElem]
    """
    Wpos=np.maximum(Wjkr,Bjkr)[:Ghosts[0,0],:] #Positived Wjk
    aux=Wpos-Bjkr[:Ghosts[0,0],:] #Vector with 0s where we positived

    corr=~np.all(aux,axis=0)&(np.any(aux<TOL, axis=0))  #Places (indices) where we need to reconstruct

    block=Wpos[:,corr] #Block of where we reconstruct
    auxblock=aux[:,corr] #Block with 0s where we reconstruct
    Bjk=Bjkr[:,corr]
    wj=Wjr[corr]
    Bj=Bjr[corr]
    xj=xjr[corr]
    yj=yjr[corr]
    xjk=xjkr[:,corr]
    yjk=yjkr[:,corr]
    Wjm=Wmjr[:,corr]
    Hjm=Hmjr[:,corr]

    xjkm = 0.5*(xjk[[0,1,2],:] + xjk[[1,2,0], :]) 
    yjkm = 0.5*(yjk[[0,1,2],:] + yjk[[1,2,0], :])
    
    for i in range(len(block[0])):
        locus=np.array(np.where(auxblock[:,i]==0)) #for debug purposes
        stay1=int(locus[0][0]) #index of first 0
        stay2=int(np.where(auxblock[:,i]==0)[0][-1]) #index of last 0
        change=list({0,1,2}.difference({stay1,stay2})) #indices where not 0
        
        Bjk1=Bjk[stay1][i]
        Bjk2=Bjk[stay2][i]*(2-len(change))+Bjk[change[0]][i]*(len(change)-1)

        xjk1=xjk[stay1][i]
        xjk2=xjk[stay2][i]*(2-len(change))+xjk[change[0]][i]*(len(change)-1)

        yjk1=yjk[stay1][i]
        yjk2=yjk[stay2][i]*(2-len(change))+yjk[change[0]][i]*(len(change)-1)

        W=(len(change)-1)*3*(wj[i] - Bj[i])/2 + Bjk2

        block[change[0]][i]=W

        #block[change[-1]][i]=(W*xjk1*yj[i] - Bjk1*xjk2*yj[i] + Bjk1*xjk[change[-1]][i]*yj[i] - W*xjk[change[-1]][i]*yj[i] - W*xj[i]*yjk1 + wj[i]*xjk2*yjk1 + W*xjk[change[-1]][i]*yjk1 - wj[i]*xjk[change[-1]][i]*yjk1 + Bjk1*xj[i]*yjk2 - wj[i]*xjk1*yjk2 - Bjk1*xjk[change[-1]][i]*yjk2 + wj[i]*xjk[change[-1]][i]*yjk2 - Bjk1*xj[i]*yjk[change[-1]][i] + W*xj[i]*yjk[change[-1]][i] - W*xjk1*yjk[change[-1]][i] + wj[i]*xjk1*yjk[change[-1]][i] + Bjk1*xjk2*yjk[change[-1]][i] - wj[i]*xjk2*yjk[change[-1]][i])/(xjk1*yj[i] - xjk2*yj[i] - xj[i]*yjk1 + xjk2*yjk1 + xj[i]*yjk2 - xjk1*yjk2)
        block[change[-1]][i]=(1/(xjk2*yjk1 - xjk1*yjk2 + (-yjk1 + yjk2)*xj[i] + (xjk1 - xjk2)*yj[i]))*(-W*yjk1*xj[i] + Bjk1*yjk2*xj[i] + W*xjk1*yj[i] - Bjk1*xjk2*yj[i] + W*yjk1*xjk[change[-1]][i] - Bjk1*yjk2*xjk[change[-1]][i] + Bjk1*yj[i]*xjk[change[-1]][i] - W*yj[i]*xjk[change[-1]][i] + (-W*xjk1 + Bjk1*xjk2 + (-Bjk1 + W)*xj[i])*yjk[change[-1]][i] + wj[i]*(xjk2*yjk1 - xjk1*yjk2 + (-yjk1 + yjk2)*xjk[change[-1]][i] + (xjk1 - xjk2)*yjk[change[-1]][i]))



        Wjm[0][i]=(W*xjk1*yj[i] - Bjk1*xjk2*yj[i] + Bjk1*xjkm[0][i]*yj[i] - W*xjkm[0][i]*yj[i] - W*xj[i]*yjk1 + wj[i]*xjk2*yjk1 + W*xjkm[0][i]*yjk1 - wj[i]*xjkm[0][i]*yjk1 + Bjk1*xj[i]*yjk2 - wj[i]*xjk1*yjk2 - Bjk1*xjkm[0][i]*yjk2 + wj[i]*xjkm[0][i]*yjk2 - Bjk1*xj[i]*yjkm[0][i] + W*xj[i]*yjkm[0][i] - W*xjk1*yjkm[0][i] + wj[i]*xjk1*yjkm[0][i] + Bjk1*xjk2*yjkm[0][i] - wj[i]*xjk2*yjkm[0][i])/(xjk1*yj[i] - xjk2*yj[i] - xj[i]*yjk1 + xjk2*yjk1 + xj[i]*yjk2 - xjk1*yjk2)
        Wjm[1][i]=(W*xjk1*yj[i] - Bjk1*xjk2*yj[i] + Bjk1*xjkm[1][i]*yj[i] - W*xjkm[1][i]*yj[i] - W*xj[i]*yjk1 + wj[i]*xjk2*yjk1 + W*xjkm[1][i]*yjk1 - wj[i]*xjkm[1][i]*yjk1 + Bjk1*xj[i]*yjk2 - wj[i]*xjk1*yjk2 - Bjk1*xjkm[1][i]*yjk2 + wj[i]*xjkm[1][i]*yjk2 - Bjk1*xj[i]*yjkm[1][i] + W*xj[i]*yjkm[1][i] - W*xjk1*yjkm[1][i] + wj[i]*xjk1*yjkm[1][i] + Bjk1*xjk2*yjkm[1][i] - wj[i]*xjk2*yjkm[1][i])/(xjk1*yj[i] - xjk2*yj[i] - xj[i]*yjk1 + xjk2*yjk1 + xj[i]*yjk2 - xjk1*yjk2)
        Wjm[2][i]=(W*xjk1*yj[i] - Bjk1*xjk2*yj[i] + Bjk1*xjkm[2][i]*yj[i] - W*xjkm[2][i]*yj[i] - W*xj[i]*yjk1 + wj[i]*xjk2*yjk1 + W*xjkm[2][i]*yjk1 - wj[i]*xjkm[2][i]*yjk1 + Bjk1*xj[i]*yjk2 - wj[i]*xjk1*yjk2 - Bjk1*xjkm[2][i]*yjk2 + wj[i]*xjkm[2][i]*yjk2 - Bjk1*xj[i]*yjkm[2][i] + W*xj[i]*yjkm[2][i] - W*xjk1*yjkm[2][i] + wj[i]*xjk1*yjkm[2][i] + Bjk1*xjk2*yjkm[2][i] - wj[i]*xjk2*yjkm[2][i])/(xjk1*yj[i] - xjk2*yj[i] - xj[i]*yjk1 + xjk2*yjk1 + xj[i]*yjk2 - xjk1*yjk2)
    
    Wjkr[:,corr]=block

    Wmjr[:,corr]=Wjm

    ind = (Wjkr - Bjkr) < TOL
    Wjkr[ind] = Bjkr[ind]

    ind = (Wmjr - Bmjr) < TOL
    Wmjr[ind] = Bmjr[ind]

    Hmjr = np.maximum(0.01*DRY,Wmjr - Bmjr)


    return Wjkr, Wmjr, Hmjr