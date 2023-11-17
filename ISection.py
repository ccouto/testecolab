# %%
import numpy as np

thetas=[20,100,200,300,400,500,600,700,800,900,1000,1100,1200]
kE_all=[1,1,0.9,0.8,0.7,0.6,0.31,0.13,0.09,0.0675,0.045,0.0225,0]
kfy_all=[1,1,1,1,1,0.78,0.47,0.23,0.11,0.06,0.04,0.02,0]
k02p_all=[1,1,0.89,0.78,0.65,0.53,0.3,0.13,0.07,0.05,0.03,0.02,0.0]
k02p_frenchannex_all=[1,1,0.896,0.793,0.694,0.557,0.318,0.15,0.078,0.048,0.032,0.016,0]

def CalcMpl20(hw,tw,b,tf,fy):
    return ((tw*(hw+tf)**2/4)+b*(hw+tf)*tf)*fy*10**(-6)

def I_Properties(h,b,tw,tf,r):
    #d = h - 2 * tf - 2 * r
    

    PI=3.141592653589793
    A = 2 * tf * b + (h - 2 * tf) * tw + (4 - PI) * r ** 2

    Iy = 1 / 12 * (b * h ** 3 - (b - tw) * (h - 2 * tf) ** 3) + 0.03 * r ** 4 + 0.2146 * r ** 2 * (h - 2 * tf - 0.4468 * r) ** 2
    Iz = 1 / 12 * (2 * tf * b ** 3 + (h - 2 * tf) * tw ** 3) + 0.03 * r ** 4 + 0.2146 * r ** 2 * (tw + 0.4468 * r) ** 2 

    iy = (Iy / A)**0.5 #
    iz = (Iz / A)**0.5 #

    Wy = 2 * Iy / h #
    Wz = 2 * Iz / b #

    Wply = tw * h ** 2 / 4 + (b - tw) * (h - tf) * tf + (4 - PI) / 2 * r ** 2 * (h - 2 * tf) + (3 * PI - 10) / 3 * r ** 3
    Wplz = b ** 2 * tf / 2 + (h - 2 * tf) / 4 * tw ** 2 + r ** 3 * (10 / 3 - PI) + (2 - PI / 2) * tw * r ** 2
    It = 2 / 3 * (b - 0.63 * tf) * tf ** 3 + 1 / 3 * (h - 2 * tf) * tw ** 3 + 2 * (tw / tf) * (0.145 + 0.1 * r / tf) * (((r + tw / 2) ** 2 + (r + tf) ** 2 - r ** 2) / (2 * r + tf)) ** 4
    Iw = tf * b ** 3 / 24 * (h - tf) ** 2
    
    return A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw

def Imono_Properties(hw,tw,b1,t1, b2, t2,r1, r2):
    PI=3.141592653589793

    h = hw+ t1 + t2

    A = b1 * t1 + b2 * t2 + (h - t1 - t2) * tw + ((4 - PI) * r1 ** 2) / 2 + ((4 - PI) * r2 ** 2) / 2

    vr1=r1*(1-(2/(3*(4-PI))))
    vr2=r2*(1-(2/(3*(4-PI))))

    Ir1=r1**4*(1/3 - PI/16 - 1/(9*(4-PI)))
    Ir2=r2**4*(1/3 - PI/16 - 1/(9*(4-PI)))
    
    zg = (1 / A) * ((b1 * t1 * (h - t1 / 2)) + (b2 * t2 * (t2 / 2)) + ((h - t1 - t2) * tw * (t2 + ((h - t1 - t2) / 2))) + (2 * (((4 - PI) * r1 ** 2) / 4) * (h - t1 - vr1)) + (2 * (((4 - PI) * r2 ** 2) / 4) * (t2 - vr2)))

    Iy = 1 / 12 * (b1 * t1 ** 3 + b2 * t2 ** 3 + tw * (h - t1 - t2) ** 3) + 2 * Ir1 + 2 * Ir2 + (b1 * t1) * ((h - t1 / 2 - zg) ** 2) + (b2 * t2) * ((t2 / 2 - zg) ** 2) + ((h - t1 - t2) * tw) * ((t2 + ((h - t1 - t2) / 2) - zg) ** 2) + 2 * (((4 - PI) * r1 ** 2) / 4) * ((h - t1 - vr1 - zg) ** 2) + 2 * (((4 - PI) * r2 ** 2) / 4) * ((t2 + vr2 - zg) ** 2)
    Iz = 1 / 12 * (t1 * b1 ** 3 + t2 * b2 ** 3 + (h - t1 - t2) * tw ** 3) + 2 * Ir1 + 2 * Ir2 + 2 * (((4 - PI) * r1 ** 2) / 4) * ((tw / 2 + vr1) ** 2) + 2 * (((4 - PI) * r2 ** 2) / 4) * ((tw / 2 + vr2) ** 2)

    iy = (Iy / A)**0.5
    iz = (Iz / A)**0.5

    Wy = min(Iy / (h - zg), Iy / zg)
    Wz = 2 * Iz / max(b1, b2)

    Wy_sup = Iy / (h - zg)
    Wy_inf = Iy / zg

    if b1 * t1 >= A / 2:
        zp = A / (2 * b1)
        zpl = zp
        Wply = (b1 * (zp ** 2) / 2) + (b1 * ((t1 - zp) ** 2) / 2) + (b2 * t2) * (h - (t2 / 2) - zp) + (tw * (h - t1 - t2)) * ((h - t1 - t2) / 2 + t1 - zp) + 2 * (((4 - PI) * r2 ** 2) / 4) * (h - t2 - vr2 - zp) + 2 * (((4 - PI) * r1 ** 2) / 4) * (t1 + vr1 - zp)
    elif b2 * t2 >= A / 2:
        zp = A / (2 * b2)
        zpl = zp
        Wply = (b2 * (zp ** 2) / 2) + (b2 * ((t2 - zp) ** 2) / 2) + (b1 * t1) * (h - (t1 / 2) - zp) + (tw * (h - t1 - t2)) * ((h - t1 - t2) / 2 + t2 - zp) + 2 * (((4 - PI) * r1 ** 2) / 4) * (h - t1 - vr1 - zp) + 2 * (((4 - PI) * r2 ** 2) / 4) * (t2 + vr2 - zp)
    else:
        zp = ((A - 2 * (((4 - PI) * r1 ** 2) / 4) - 2 * (((4 - PI) * r2 ** 2) / 4)) / 2 - (b2 * t2)) / tw + t2
        zpl = zp
        Wply = (tw * ((zp - t2) ** 2) / 2) + (tw * ((h - t1 - zp) ** 2) / 2) + (b1 * t1) * (h - (t1 / 2) - zp) + (b2 * t2) * (zp - (t2 / 2)) + 2 * (((4 - PI) * r1 ** 2) / 4) * (h - t1 - vr1 - zp) + 2 * (((4 - PI) * r2 ** 2) / 4) * (zp - t2 - vr2)

    Wplz = ((b1 * t1 * b1) / 4) + ((b2 * t2 * b2) / 4) + ((((h - t1 - t2) * tw) * tw) / 4) + (2 * (((4 - PI) * r1 ** 2) / 4) * ((tw / 2) - vr1)) + (2 * (((4 - PI) * r2 ** 2) / 4) * ((tw / 2) - vr2))

    It1=1.0/3.0 * b1 * t1 ** 3 * (1-(t1/b1)*(0.633-0.055*(t1**3)/(b1**3)))
    It2=1.0/3.0 * b2 * t2 ** 3 * (1-(t2/b2)*(0.633-0.055*(t2**3)/(b2**3)))
    Itw = 1 / 3 * h * (tw ** 3) * (1 - (tw / h) * (0.633 - 0.055 * (tw ** 3 / h ** 3)))
    eIt4 = min(tw, t1)
    bIt4 = max(tw, t1)
    It4 = 1 / 3 * bIt4 * (eIt4 ** 3) * (1 - (eIt4 / bIt4) * (0.633 - 0.055 * (eIt4 ** 3 / bIt4 ** 3)))
    eIt5 = min(tw, t2)
    bIt5 = max(tw, t2)
    It5 = 1 / 3 * bIt5 * (eIt5 ** 3) * (1 - (eIt5 / bIt5) * (0.633 - 0.055 * (eIt5 ** 3 / bIt5 ** 3)))
    eIt6 = min(tw + 0.4 * r1, t1 + 0.4 * r1)
    bIt6 = max(tw + 0.4 * r1, t1 + 0.4 * r1)
    It6 = 1 / 3 * bIt6 * (eIt6 ** 3) * (1 - (eIt6 / bIt6) * (0.633 - 0.055 * (eIt6 ** 3 / bIt6 ** 3)))
    eIt7 = min(tw + 0.4 * r2, t2 + 0.4 * r2)
    bIt7 = max(tw + 0.4 * r2, t2 + 0.4 * r2)
    It7 = 1 / 3 * bIt7 * (eIt7 ** 3) * (1 - (eIt7 / bIt7) * (0.633 - 0.055 * (eIt7 ** 3 / bIt7 ** 3)))


    if r1 == 0:
        alfa1 = 0
    else:
        if (6 * tw + t1) <= b1:
            alfa1 = 4
        else:
            alfa1 = 8 / (1 + ((6 * tw + t1) / b1))

    if r2 == 0:
        alfa2 = 0
    else:
        if (6 * tw + t2) <= b2:
            alfa2 = 4
        else:
            alfa2 = 8 / (1 + ((6 * tw + t2) / b2))

    It = It1 + It2 + Itw - ((tw / b1) ** 2) * It4 - ((tw / b2) ** 2) * It5 + alfa1 * (It6 - It4) + alfa2 * (It7 - It5)
    Iw = 1 / 12 * (t1 * b1 ** 3) * (1 - ((1 / 12 * (t1 * b1 ** 3)) / Iz)) * ((h - t1 / 2 - t2 / 2) ** 2)


    return A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw, zg

def CalcMcr(h,tw,b,tf,r,L, temperature):
    A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw = I_Properties(h,b,tw,tf,r)
    
    E=210000 #MPa, because all dimensions are in mm
    
    C1=1.0
    C2=0.0
    C3=0.0
    kz=1.0
    kw=1.0
    zg=0.0
    zj=0.0
    
    PI=3.141592653589793
    Mcr = C1 * PI ** 2 * E * Iz / (kz * L) ** 2 * (((kz / kw) ** 2 * Iw / Iz + (kz * L) ** 2 * E / (2 * (1 + 0.3)) * It / (PI ** 2 * E * Iz) + (C2 * zg - C3 * zj) ** 2) ** 0.5 - (C2 * zg - C3 * zj))
    
    kE=np.interp(temperature,thetas,kE_all)
    
    return Mcr*kE

def CalcMcr_Pub119(hw,tw,bf1,tf1, bf2, tf2,r,psi, L, temperature):
    A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw, zg = Imono_Properties(hw,tw,bf1,tf1,bf2,tf2,r,r)
    
    E=210000 #MPa, because all dimensions are in mm

    kz=1.0
    kw=1.0
    
    zj=0.0


    #C1
    C1=min(2.6, 1.77-1.04*psi+0.27*psi**2)  #Eq. (299)
    #We are going to use interpolation
    C1_interp_1=[2.6,2.6,2.35,2.06,1.77,1.52,1.31,1.14,1]
    C1_interp_05=[2.45,2.45, 2.42, 2.15, 1.86, 1.6, 1.37, 1.19, 1.05]
    C1_psi=[-1,-3/4,-1/2,-1/4,0,1/4,1/2,3/4,1]
    C1_1=np.interp(psi,C1_psi,C1_interp_1)
    C1_05=np.interp(psi,C1_psi,C1_interp_05)
    C1=np.interp(kz,[0.5,1],[C1_05,C1_1])

    C2=0.0
    C3=0.0

    PI=3.141592653589793

    if bf1!=bf2 or tf1!=tf2:
        #mono symmetric section:        
        Ifc = tf1 * bf1**3 / 12
        Ift = tf2 * bf2**3 / 12
        psif = (Ifc-Ift)/(Ifc+Ift)
        hs = hw + tf1/2 + tf2/2
        Iw = (1-psif**2)*Iz*(hs/2)**2
        if psif>=0:
            zj = 0.8*psif*hs/2
        else:
            zj = psif*hs/2
        if psi>=0:
            C3 = 1
        else:
            C3_interp_psi=[-1,-3/4,-1/2,-1/4]
            if psif>0:
                C3_interp_values_1=[-psif,0.55-psif,1.3-1.2*psif, 0.85]
                C3_interp_values_05=[-0.125-0.7*psif,0.35-psif,0.77-psif,0.65]
            else:
                C3_interp_values_1=[-psif,1,1,1]
                C3_interp_values_05=[0.125-0.7*psif,0.85,0.95,1]
            C3_1=np.interp(psi,C3_interp_psi,C3_interp_values_1)
            C3_05=np.interp(psi,C3_interp_psi,C3_interp_values_05)
            C3=np.interp(kz,[0.5,1],[C3_05,C3_1])
            


    
    Mcr = C1 * PI ** 2 * E * Iz / (kz * L) ** 2 * (((kz / kw) ** 2 * Iw / Iz + (kz * L) ** 2 * E / (2 * (1 + 0.3)) * It / (PI ** 2 * E * Iz) + (C2 * zg - C3 * zj) ** 2) ** 0.5 - (C2 * zg - C3 * zj))
    
    kE=np.interp(temperature,thetas,kE_all)
    
    return Mcr*kE

def CalcMcr_EN1993_1_103(hw,tw,bf1,tf1, bf2, tf2,r,psi, L, temperature):
    A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw, zg = Imono_Properties(hw,tw,bf1,tf1,bf2,tf2,r,r)
    
    kz=1.0
    kw=1.0

    C2=0    #only endmoments here!

    E=210000 #MPa, because all dimensions are in mm
    G=E/(2*(1+0.3)) #MPa, shear modulus
    PI=3.141592653589793

    C10_interp=[2.555,2.547, 2.331,2.047,1.770,1.522,1.312, 1.139, 1]
    C11_interp=[2.733,2.852,2.591, 2.207,1.847, 1.551,1.320,1.141,1]
    C1_psi=[-1,-3/4,-1/2,-1/4,0,1/4,1/2,3/4,1]
    C10=np.interp(psi,C1_psi,C10_interp)
    C11=np.interp(psi,C1_psi,C11_interp)

    Ifc = tf1 * bf1**3 / 12
    Ift = tf2 * bf2**3 / 12
    psif = (Ifc-Ift)/(Ifc+Ift)

    hs = hw + tf1/2 + tf2/2

    if bf1!=bf2 or tf1!=tf2:
        #mono symmetric section:        
        # Ifc = tf1 * bf1**3 / 12
        # Ift = tf2 * bf2**3 / 12
        # psif = (Ifc-Ift)/(Ifc+Ift)
        
        Iw = (1-psif**2)*Iz*(hs/2)**2

        if psi>=0:
            C3 = 1
        else:
            C3_interp_psi=[-1,-3/4,-1/2,-1/4]
            if psif>0:
                C3_interp_values_1=[-psif,0.55-psif,1.3-1.2*psif, 0.85]
            else:
                C3_interp_values_1=[-psif,1,1,1]
            C3=np.interp(psi,C3_interp_psi,C3_interp_values_1)
    else:
        C3=0
        C2=0
    
    kwt = PI/(kw*L)*(E*Iw/(G*It))**0.5
    
    C1 = min(C10+(C11-C10)*kwt,C11)
    if kwt==0:
        C1=C10
    if kwt>=1:
        C1=C11

    zetag=0 #only endmoments here

    zj = 0.45*psif*hs


    zetaj = PI*zj/(kz*L)*(E*Iz/(G*It))**0.5

    #print('psif:',psif)

    ucr = C1/kz*(  (1+kwt**2+(C2*zetag-C3*zetaj)**2)**0.5 - (C2*zetag-C3*zetaj)  )

    Mcr = ucr * PI*(E*Iz*G*It)**0.5/L

    kE=np.interp(temperature,thetas,kE_all)
    
    return Mcr*kE

def CalcMcr_EN1993_1_103_old(h,tw,b,tf,r,Lf,psi, temperature):
    A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw = I_Properties(h,b,tw,tf,r)

    #print(It,Iw, Iz)


    E=210000 #MPa, because all dimensions are in mm
    G=E/(2*(1+0.3)) #MPa, shear modulus
    PI=3.141592653589793
    C1=1.0
    C2=0.0


    #beams with end bending moments
    C1 = min(2.3,1.75-1.05*psi+(0.3*psi**2))

    S=(E*Iw/(G*It))**0.5

    C = PI * C1 * ( (1+((PI**2*S**2/Lf**2)*(C2**2+1)))**0.5+PI*C2*S/Lf) #Eq. 5.101

    M_crM = C / Lf * (E*Iz*G*It)**0.5       #Eq. (5.3)

    kE=np.interp(temperature,thetas,kE_all)
    
    return M_crM*kE

def CalcNcr_z(h,tw,b,tf,r,Lf, temperature):
    A, Iy, Iz, iy, iz, Wy, Wz, Wply, Wplz, It, Iw = I_Properties(h,b,tw,tf,r)

    #print(It,Iw, Iz)


    E=210000 #MPa, because all dimensions are in mm
    G=E/(2*(1+0.3)) #MPa, shear modulus
    PI=3.141592653589793

    N_cr_z = PI**2*E*Iz/(Lf**2)

    kE=np.interp(temperature,thetas,kE_all)
    
    return N_cr_z*kE

if __name__ == "__main__":
    # execute only if run as a script
    #print(CalcMcr_Pub119(870-2*10,6,400,10,400,10,0,-0.333,10000,20)*10**-6)

    print(CalcMcr_Pub119(1000,5,200,12,200,8,0,0,3000,20)*10**-6)

    print(CalcMcr_Pub119(1000,5,200,8,200,12,0,0,3000,20)*10**-6)

    print(CalcMcr_EN1993_1_103(1000,5,200,12,200,8,0,0,3000,20)*10**-6)

    print(CalcMcr_EN1993_1_103(1000,5,200,8,200,12,0,0,3000,20)*10**-6)


    #print(CalcMcr_EN1993_1_103(1000,5,200,8,400,12,0,-1,6000,20)*10**-6)
    print(CalcMcr_EN1993_1_103(278.6,7.1,150,10.7,72,10.7,0,-1,6000,20)*10**-6)
    print(CalcMcr_Pub119(278.6,7.1,150,10.7,72,10.7,0,-1,6000,20)*10**-6)
    #exit()

    fOut=open('results_toMB.csv','w')
    for psi in [1,0.5,0,-0.5,-0.75,-1]:
        for L in range(3,9):
            mcr1=CalcMcr_EN1993_1_103(278.6,7.1,150,10.7,72,10.7,0,psi,L*1000,20)*10**-6
            mcr2=CalcMcr_Pub119(278.6,7.1,150,10.7,72,10.7,0,psi,L*1000,20)*10**-6
            print(L,mcr1,mcr2,file=fOut)
    fOut.close()

    exit()
    for j in range(0,10):
        print(CalcMcr_EN1993_1_103(680,10,150,10,0,1000+j*1000,-1,20)*10**-6)

    # import pandas as pd
    # df=pd.read_excel('Mcr_from_formulae.xlsx')

    # %%
    import pandas as pd
    df=pd.read_excel('Mcr_taper.xlsx',skiprows=0)
    # %%
    results=[]
    for j in range(1,len(df.hw1)):
        mcr=CalcMcr_EN1993_1_103(df.hw1[j]+2*df.tf[j],df.tw[j],df.bf[j],df.tf[j],0,df.L[j],1,20)*10**-6
        mcr2=CalcMcr_EN1993_1_103(df.hw2[j]+2*df.tf[j],df.tw[j],df.bf[j],df.tf[j],0,df.L[j],1,20)*10**-6
        results.append([mcr,mcr2])
        #print(j, mcr, mcr2)
    np.savetxt('results_of_mcr_taper.csv',results,delimiter=";")
    print("Done!")
    # %%
    results=[]
    for j in range(1,len(df.hw1)):
        ncr=CalcNcr_z(df.hw1[j]+2*df.tf[j],df.tw[j],df.bf[j],df.tf[j],0,df.L[j],20)*10**-3
        results.append(ncr)
        print(j)
    np.savetxt('results_of_ncr_z.csv',results)


# %%
    mcr=CalcMcr_EN1993_1_103(862.4,31.2,450,31.2,0,10400,1,20)*10**-6
    print(mcr)
# %%
