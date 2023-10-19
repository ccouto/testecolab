import pickle
import numpy as np
import pyrenn_ccversion as prn

def Mcr(h,tw,b,tf,r,L):
    Iz = 1 / 12 * (2 * tf * b ** 3 + (h - 2 * tf) * tw ** 3) + 0.03 * r ** 4 + 0.2146 * r ** 2 * (tw + 0.4468 * r) ** 2 
    It = 2 / 3 * (b - 0.63 * tf) * tf ** 3 + 1 / 3 * (h - 2 * tf) * tw ** 3 + 2 * (tw / tf) * (0.145 + 0.1 * r / tf) * (((r + tw / 2) ** 2 + (r + tf) ** 2 - r ** 2) / (2 * r + tf)) ** 4
    Iw = tf * b ** 3 / 24 * (h - tf) ** 2
   
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
     
    return Mcr

def LoadNN(nn_file, verbose=False):

    nnfolder=''#nn_file.rpartition('/')[0]+'/'
    nn=nn_file.rpartition('/')[2][:-3]
    scaler_file=nnfolder+'scaler122.pkl'    
    f=open(nnfolder+'scaler_y.txt')
    u=float(f.readline())
    
    f.close()
    
    model_scaler=pickle.load(open(scaler_file,'rb'))

    try:
        model_scaler=pickle.load(open(scaler_file,'rb'))
        if verbose: print("Scaler ok.")
    except:
        #print("there was an error opening the model scaler...")
        if verbose: print("Could not open model scaler!")
    model=prn.loadNN(nn_file)

    if verbose:
        print('folder is',nnfolder, 'and name of nn is','"'+nn+'"')
        print('scaler file is',scaler_file)
        print('scaler y max:', u)

    return model, model_scaler, u

def MakeNNRangeChecks(hw1,hw2,tw,bf1,tf1,bf2,tf2,L):
    warn_msg=''
    #note, hw2 is always greater than hw1 when this function is called
    #
    #Table 2 of Couto (2022) paper
    if max(hw2,hw1)/min(hw1,hw2)>4: warn_msg=f'hw,max/hw,min={max(hw2,hw1)/min(hw1,hw2):.3f} > 4'
    if bf1/bf2<0.25: warn_msg=f'bf1/bf2={bf1/bf2:.3f} < 0.25'
    if bf1/bf2>4: warn_msg=f'bf1/bf2={bf1/bf2:.3f} > 4'
    if hw2/max(bf1,bf2)<1: warn_msg=f'hw,max/bf,max={hw2/max(bf1,bf2):.3f} < 1'
    if hw2/max(bf1,bf2)>4: warn_msg=f'hw,max/bf,max={hw2/max(bf1,bf2):.3f} > 4'
    if L/max(hw2,hw1)<2: warn_msg=f'L/hw,max={L/max(hw2,hw1):.3f} < 2'
    if L/max(hw2,hw1)>40: warn_msg=f'L/hw,max={L/max(hw2,hw1):.3f} > 40'
    if bf1/tf1<6.25: warn_msg=f'bf1/tf1={bf1/tf1:.3f} < 6.25'
    if bf1/tf1>100: warn_msg=f'bf1/tf1={bf1/tf1:.3f} > 100'
    if bf2/tf2<6.25: warn_msg=f'bf2/tf2={bf2/tf2:.3f} < 6.25'
    if bf2/tf2>100: warn_msg=f'bf2/tf2={bf2/tf2:.3f} > 100'
    if hw2/tw<25: warn_msg=f'hw,max/tw={hw2/tw:.3f} < 25'  
    if hw2/tw>300: warn_msg=f'hw,max/tw={hw2/tw:.3f} > 300'
    if tf1/tw<1: warn_msg=f'tf1/tw={tf1/tw:.3f} < 1'
    if tf1/tw>24: warn_msg=f'tf1/tw={tf1/tw:.3f} > 24'
    if tf2/tw<1: warn_msg=f'tf2/tw={tf2/tw:.3f} < 1'
    if tf2/tw>24: warn_msg=f'tf2/tw={tf2/tw:.3f} > 24'

    # NOTE: if you want to check results of NN outside the scope change next line uncomment next line
    # return True

    #all checks are ok!
    if warn_msg!='':
        print(f"Ratio {warn_msg} is outside scope of application. \n\nNo results are calculated.")
        #
        return False
    else:
        return True

def CalcMcr_NN(hw1, hw2,tw,bf1,tf1,bf2,tf2,psi, L, model, model_scaler,u):
    
    
    x=np.zeros(9)

    mcr_u=Mcr((hw1+hw2)/2+tf1+tf2,tw,(bf1+bf2)/2,(tf1+tf2)/2,0,L)
    
    x[0]=hw1
    x[1]=hw2
    x[2]=tw
    x[3]=bf1
    x[4]=tf1
    x[5]=bf2
    x[6]=tf2
    x[7]=psi
    x[8]=L
    

    case = np.array([[x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]]])
    x_case=model_scaler.transform(case)

    result=prn.NNOut(x_case.transpose(),model).sum()
    scale_u=u

    return result*scale_u*mcr_u

def Calculate(hw1,hw2, tw, bf1, tf1, bf2,tf2, L, M1, M2, model, model_scaler, u):

        MMax=max(abs(M1),abs(M2))

        swap_flanges=False
        if MMax==abs(M1):
            #side 1 is hw1
            hw1=hw1
            hw2=hw2
            if float(M1)<0:
                swap_flanges=True
                #change bending signs
                M1*=-1
                M2*=-1
        else:
            #side 1 is hw2
            tmp=hw1
            hw1=hw2
            hw2=tmp
            if float(M2)<0:
                swap_flanges=True
                #change bending signs
                M1*=-1
                M2*=-1
            #change the bending moments
            tmp=M1
            M1=M2
            M2=tmp


        

        if swap_flanges:
            #swap the flanges
            bf_temp=bf2
            tf_temp=tf2
            bf2=bf1
            tf2=tf1
            bf1=bf_temp
            tf1=tf_temp

            
        the_min=min(float(M1),float(M2))/MMax
        the_max=max(float(M1),float(M2))/MMax

        psi=the_min/the_max

        #check the ranges of input values and warn the user if outside the range
        if not MakeNNRangeChecks(float(hw1),float(hw2),float(tw),float(bf1),float(tf1),float(bf2),float(tf2),float(L)):
            # self.Results.configure(text='No results.', justify='left')
            return


            
            
            

        #if float(M2)==0 or float(M1)==0:
        #    psi=0
        #else:
        #    psi=float(M1)/MMax/(1*(float(M2)/MMax))

        #print(hw2)
     
        #msg.showinfo('Inputs','hw1='+str(hw1)+'\n'+'hw2='+str(hw2))
        #self.Results.configure(text='teste')
        #self.Results.configure(fg='#FF0000')
        #self.hw1.configure(justify='left')

        
        Mcr=CalcMcr_NN(float(hw1),float(hw2),float(tw),float(bf1),float(tf1),float(bf2),float(tf2),psi,float(L),model,model_scaler,u)*10**-6
        
        txt_result='alphacr= '+str(round(Mcr/MMax,3))
        txt_result+='\n\nMcr= '+str(round(Mcr,3))+' kN.m'
        txt_result+='\n\npsi='+str(psi)
        #txt_result+='\n\ncalculation time was '+str(round(end-start,5))+' seconds'
        print('Results:\n\n'+txt_result)