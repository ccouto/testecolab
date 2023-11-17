from logging import exception
import tkinter as tk
import tkinter.font as tkFont
import tkinter.messagebox as msg
from PIL import ImageTk, Image
import PIL.ImageTk
import PIL.Image

#measuring time
from timeit import default_timer as timer

# %%
import numpy as np
import pyrenn_ccversion as prn
import pickle
from ISection import CalcMcr as CalcMcr_unif
import os
#import pandas as pd

#import warnings
#warnings.filterwarnings("ignore")

#%%
def LoadNN(nn_file, verbose=False):

    nnfolder=nn_file.rpartition('/')[0]+'/'
    nn=nn_file.rpartition('/')[2][:-3]
    scaler_file=nnfolder+'scaler.pkl'    
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


def CalcMcr_NN(hw1, hw2,tw,bf1,tf1,bf2,tf2,psi, L, model, model_scaler,u):
    
    #print(Mpl20)
    x=np.zeros(9)

    mcr_u=CalcMcr_unif((hw1+hw2)/2+tf1+tf2,tw,(bf1+bf2)/2,(tf1+tf2)/2,0,L,20)
    #mcr_u2=CalcMcr_unif(hw2+2*tf,tw,b,tf,0,hw2+2*tf,20)

    x[0]=hw1
    x[1]=hw2
    x[2]=tw
    x[3]=bf1
    x[4]=tf1
    x[5]=bf2
    x[6]=tf2
    x[7]=psi
    x[8]=L
    #x[7]=mcr_u/mcr_u2
    
    #n_features=7


    case = np.array([[x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]]])
    x_case=model_scaler.transform(case)

    #print(case)

    #x_case=model_scaler.transform(case)
    result=prn.NNOut(x_case.transpose(),model).sum()
    scale_u=u
    #print(result*mx/mx)


    #mcr_u=CalcMcr_unif(hw1+2*tf1,tw,bf1,tf1,0,L,20)
    mcr_u=CalcMcr_unif((hw1+hw2)/2+tf1+tf2,tw,(bf1+bf2)/2,(tf1+tf2)/2,0,L,20)
    #print('Mpl20', Mpl20, 'y_pred:', result)
    
    #return result*scale_u
    return result*scale_u*mcr_u

def Calcheq(hw1, hw2,tw,bf1,tf1,bf2,tf2,psi, L, model, model_scaler,u):
    
    #print(Mpl20)
    x=np.zeros(9)

    mcr_u=CalcMcr_unif((hw1+hw2)/2+tf1+tf2,tw,(bf1+bf2)/2,(tf1+tf2)/2,0,L,20)
    #mcr_u2=CalcMcr_unif(hw2+2*tf,tw,b,tf,0,hw2+2*tf,20)

    x[0]=hw1
    x[1]=hw2
    x[2]=tw
    x[3]=bf1
    x[4]=tf1
    x[5]=bf2
    x[6]=tf2
    x[7]=psi
    x[8]=L
    #x[7]=mcr_u/mcr_u2
    
    #n_features=7


    case = np.array([[x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]]])
    x_case=model_scaler.transform(case)

    #print(case)

    #x_case=model_scaler.transform(case)
    result=prn.NNOut(x_case.transpose(),model).sum()
    scale_u=u
    #print(result*mx/mx)


    #mcr_u=CalcMcr_unif(hw1+2*tf1,tw,bf1,tf1,0,L,20)
    mcr_u=CalcMcr_unif((hw1+hw2)/2+tf1+tf2,tw,(bf1+bf2)/2,(tf1+tf2)/2,0,L,20)
    #print('Mpl20', Mpl20, 'y_pred:', result)
    
    #return result*scale_u
    Mcr_orig=result*scale_u*mcr_u
    Mcr_target=0
    hw_eq=min(hw1,hw2)
    step=(max(hw1,hw2)-min(hw1,hw2))/10

    while(abs(Mcr_orig-Mcr_target)>0.1):
        return 0





def importimg(file_name):
    img = PIL.Image.open(file_name)
    newimg=img.resize((746,223))
    return PIL.ImageTk.PhotoImage(newimg)

def is_str_a_float(val):
    try:
        m=float(val)
        return True
    except:
        return False


class App:
    


    def __init__(self, root):
        #global GLineEdit_260
        #setting title
        root.title("Elastic critical moment with neural networks")
        #setting window size
        width=746
        height=600
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        root.geometry(alignstr)
        root.resizable(width=False, height=False)

        posy=260
        GLabel_777=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_777["font"] = ft
        GLabel_777["fg"] = "#333333"
        GLabel_777["justify"] = "center"
        GLabel_777["text"] = "hw1 (mm)"
        GLabel_777.place(x=20,y=posy,width=112,height=30)
        self.hw1=tk.Entry(root)
        self.hw1["borderwidth"] = "1px"
        self.hw1["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.hw1["font"] = ft
        self.hw1["fg"] = "#333333"
        self.hw1["justify"] = "center"
        self.hw1["text"] = "hw1"
        self.hw1["relief"] = "flat"
        self.hw1.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_778=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_778["font"] = ft
        GLabel_778["fg"] = "#333333"
        GLabel_778["justify"] = "center"
        GLabel_778["text"] = "hw2 (mm)"
        GLabel_778.place(x=20,y=posy,width=112,height=30)
        self.hw2=tk.Entry(root)
        self.hw2["borderwidth"] = "1px"
        self.hw2["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.hw2["font"] = ft
        self.hw2["fg"] = "#333333"
        self.hw2["justify"] = "center"
        self.hw2["text"] = "hw2"
        self.hw2["relief"] = "flat"
        self.hw2.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_tw=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_tw["font"] = ft
        GLabel_tw["fg"] = "#333333"
        GLabel_tw["justify"] = "center"
        GLabel_tw["text"] = "tw (mm)"
        GLabel_tw.place(x=20,y=posy,width=112,height=30)
        self.tw=tk.Entry(root)
        self.tw["borderwidth"] = "1px"
        self.tw["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.tw["font"] = ft
        self.tw["fg"] = "#333333"
        self.tw["justify"] = "center"
        self.tw["text"] = "tw"
        self.tw["relief"] = "flat"
        self.tw.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_779=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_779["font"] = ft
        GLabel_779["fg"] = "#333333"
        GLabel_779["justify"] = "center"
        GLabel_779["text"] = "bf1 (mm)"
        GLabel_779.place(x=20,y=posy,width=112,height=30)
        self.bf1=tk.Entry(root)
        self.bf1["borderwidth"] = "1px"
        self.bf1["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.bf1["font"] = ft
        self.bf1["fg"] = "#333333"
        self.bf1["justify"] = "center"
        self.bf1["text"] = "bf1"
        self.bf1["relief"] = "flat"
        self.bf1.place(x=140,y=posy,width=70,height=25)
        
        posy+=30
        GLabel_780=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_780["font"] = ft
        GLabel_780["fg"] = "#333333"
        GLabel_780["justify"] = "center"
        GLabel_780["text"] = "tf1 (mm)"
        GLabel_780.place(x=20,y=posy,width=112,height=30)
        self.tf1=tk.Entry(root)
        self.tf1["borderwidth"] = "1px"
        self.tf1["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.tf1["font"] = ft
        self.tf1["fg"] = "#333333"
        self.tf1["justify"] = "center"
        self.tf1["text"] = "tf1"
        self.tf1["relief"] = "flat"
        self.tf1.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_781=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_781["font"] = ft
        GLabel_781["fg"] = "#333333"
        GLabel_781["justify"] = "center"
        GLabel_781["text"] = "bf2 (mm)"
        GLabel_781.place(x=20,y=posy,width=112,height=30)
        self.bf2=tk.Entry(root)
        self.bf2["borderwidth"] = "1px"
        self.bf2["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.bf2["font"] = ft
        self.bf2["fg"] = "#333333"
        self.bf2["justify"] = "center"
        self.bf2["text"] = "bf2"
        self.bf2["relief"] = "flat"
        self.bf2.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_782=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_782["font"] = ft
        GLabel_782["fg"] = "#333333"
        GLabel_782["justify"] = "center"
        GLabel_782["text"] = "tf2 (mm)"
        GLabel_782.place(x=20,y=posy,width=112,height=30)
        self.tf2=tk.Entry(root)
        self.tf2["borderwidth"] = "1px"
        self.tf2["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.tf2["font"] = ft
        self.tf2["fg"] = "#333333"
        self.tf2["justify"] = "center"
        self.tf2["text"] = "tf2"
        self.tf2["relief"] = "flat"
        self.tf2.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_L=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_L["font"] = ft
        GLabel_L["fg"] = "#333333"
        GLabel_L["justify"] = "center"
        GLabel_L["text"] = "L (mm)"
        GLabel_L.place(x=20,y=posy,width=112,height=30)
        self.L=tk.Entry(root)
        self.L["borderwidth"] = "1px"
        self.L["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.L["font"] = ft
        self.L["fg"] = "#333333"
        self.L["justify"] = "center"
        self.L["text"] = "L"
        self.L["relief"] = "flat"
        self.L.place(x=140,y=posy,width=70,height=25)

        posy+=30
        GLabel_M1=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_M1["font"] = ft
        GLabel_M1["fg"] = "#333333"
        GLabel_M1["justify"] = "center"
        GLabel_M1["text"] = "M1 (kN.m)"
        GLabel_M1.place(x=20,y=posy,width=112,height=30)
        self.M1=tk.Entry(root)
        self.M1["borderwidth"] = "1px"
        self.M1["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.M1["font"] = ft
        self.M1["fg"] = "#333333"
        self.M1["justify"] = "center"
        self.M1["text"] = "M1"
        self.M1["relief"] = "flat"
        self.M1.place(x=140,y=posy,width=70,height=25)

        GLabel_M1sign=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_M1sign["font"] = ft
        GLabel_M1sign["fg"] = "#333333"
        GLabel_M1sign["justify"] = "center"
        GLabel_M1sign["text"] = "(M1 > 0 produces compression in upper flange 1)"
        GLabel_M1sign.place(x=210,y=posy,width=312,height=30)

        posy+=30
        GLabel_M2=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_M2["font"] = ft
        GLabel_M2["fg"] = "#333333"
        GLabel_M2["justify"] = "center"
        GLabel_M2["text"] = "M2 (kN.m)"
        GLabel_M2.place(x=20,y=posy,width=112,height=30)
        self.M2=tk.Entry(root)
        self.M2["borderwidth"] = "1px"
        self.M2["cursor"] = "xterm"
        ft = tkFont.Font(family='Times',size=10)
        self.M2["font"] = ft
        self.M2["fg"] = "#333333"
        self.M2["justify"] = "center"
        self.M2["text"] = "M2"
        self.M2["relief"] = "flat"
        self.M2.place(x=140,y=posy,width=70,height=25)

        GLabel_M2sign=tk.Label(root)
        ft = tkFont.Font(family='Times',size=10)
        GLabel_M2sign["font"] = ft
        GLabel_M2sign["fg"] = "#333333"
        GLabel_M2sign["justify"] = "center"
        GLabel_M2sign["text"] = "(M2 > 0 produces compression in upper flange 1)"
        GLabel_M2sign.place(x=210,y=posy,width=312,height=30)

        if os.path.exists(os.path.expanduser('~/mcrnet')):
            #read saved file

            out = open(os.path.expanduser('~/mcrnet'),'r')
            values=out.readlines()
            self.hw1.insert(0,values[0].replace('\n',''))
            self.hw2.insert(0,values[1].replace('\n',''))
            self.tw.insert(0,values[2].replace('\n',''))
            self.L.insert(0,values[3].replace('\n',''))
            self.bf1.insert(0,values[4].replace('\n',''))
            self.bf2.insert(0,values[5].replace('\n',''))
            self.tf1.insert(0,values[6].replace('\n',''))
            self.tf2.insert(0,values[7].replace('\n',''))
            self.M1.insert(0,values[8].replace('\n',''))
            self.M2.insert(0,values[9].replace('\n',''))

            out.close()

        else:
            self.hw1.insert(0,'450')
            self.hw2.insert(0,'650')
            self.tw.insert(0,'4')
            self.bf1.insert(0,'180')
            self.tf1.insert(0,'10')
            self.bf2.insert(0,'180')
            self.tf2.insert(0,'10')
            self.L.insert(0,'10000')
            self.M1.insert(0,'1')
            self.M2.insert(0,'-1')
        
        
        
        #self.hw2.set('30')



        # self.WarnLabel=tk.Label(root,anchor="w", justify='left')
        # ft = tkFont.Font(family='Times',size=10)
        # self.WarnLabel["font"] = ft
        # self.WarnLabel["fg"] = "#f00"
        # self.WarnLabel["justify"] = "left"
        # self.WarnLabel["text"] = ""
        # self.WarnLabel.place(x=250,y=320,width=425,height=50)
        


        self.Results=tk.Label(root, anchor="w", justify='left')
        ft = tkFont.Font(family='Times',size=10)
        self.Results["font"] = ft
        self.Results["fg"] = "#333333"
        self.Results["justify"] = "left"
        self.Results["text"] = ""

        self.Results.place(x=250,y=260,width=325,height=140)

        #self.WarnLabel.tkraise(self.Results)

        GButton_84=tk.Button(root)
        GButton_84["bg"] = "#efefef"
        ft = tkFont.Font(family='Times',size=10)
        GButton_84["font"] = ft
        GButton_84["fg"] = "#000000"
        GButton_84["justify"] = "center"
        GButton_84["text"] = "Calculate"
        GButton_84.place(x=600,y=520,width=70,height=25)
        GButton_84["command"] = self.GButton_84_command

    
        bg_img = importimg('notation.png')
        
        self.bg = tk.Label(root, image=bg_img)
        self.bg.image = bg_img  #update the image in the label background
        self.bg.grid(column=0, row=0)        
    
    def MakeInputChecks(self):
        invalid_label=''
        if not is_str_a_float(self.hw1.get()): invalid_label='hw1'
        if not is_str_a_float(self.hw2.get()): invalid_label='hw2'
        if not is_str_a_float(self.tw.get()): invalid_label='tw'
        if not is_str_a_float(self.L.get()): invalid_label='L'
        if not is_str_a_float(self.bf1.get()): invalid_label='bf1'
        if not is_str_a_float(self.bf2.get()): invalid_label='bf2'
        if not is_str_a_float(self.tf1.get()): invalid_label='tf1'
        if not is_str_a_float(self.tf2.get()): invalid_label='tf2'

        
        if not is_str_a_float(self.M1.get()): invalid_label='M1'
        if not is_str_a_float(self.M2.get()): invalid_label='M2'


        #all checks are ok!
        if invalid_label=='':
            return True
        else:
            msg.showerror(f"Error!",f"Invalid value for {invalid_label}. Please define a numeric value!")
            return False

    def MakeNNRangeChecks(self,hw1,hw2,tw,bf1,tf1,bf2,tf2,L):
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
            self.Results.configure(text=f"Ratio {warn_msg} is outside scope of application. \n\nNo results are calculated.", fg='red')
            #
            return False

            #msg.showwarning(f"Warning!",f"Ratio {warn_msg} is outside the training range of the neural network model. \n\nPlease proceed with caution since the neural network model may predict erroneous results.")
        else:
            self.Results.configure(fg='black')
            #self.WarnLabel.configure(text="")
            return True

    def SaveFile(self):
        try:
            out = open(os.path.expanduser('~/mcrnet'),'w')
            print(self.hw1.get(),file=out)
            print(self.hw2.get(),file=out)
            print(self.tw.get(),file=out)
            print(self.L.get(),file=out)
            print(self.bf1.get(),file=out)
            print(self.bf2.get(),file=out)
            print(self.tf1.get(),file=out)
            print(self.tf2.get(),file=out)
            print(self.M1.get(),file=out)
            print(self.M2.get(),file=out)
            out.close()
            return
        finally:
            return

    def GButton_84_command(self):

        if not self.MakeInputChecks():
            return

        self.SaveFile()

        tw=str(self.tw.get())
        M1=float(str(self.M1.get()))
        M2=float(str(self.M2.get()))

        MMax=max(abs(M1),abs(M2))

        reverse_flanges=False
        if MMax==abs(M1):
            #side 1 is hw1
            hw1=str(self.hw1.get())
            hw2=str(self.hw2.get())
            if float(M1)<0:
                reverse_flanges=True
                #change bending signs
                M1*=-1
                M2*=-1
        else:
            #side 1 is hw2
            hw1=str(self.hw2.get())
            hw2=str(self.hw1.get())
            if float(M2)<0:
                reverse_flanges=True
                #change bending signs
                M1*=-1
                M2*=-1
            #change the bending moments
            tmp=M1
            M1=M2
            M2=tmp


        L=str(self.L.get())

        if not reverse_flanges:
            bf1=str(self.bf1.get())
            tf1=str(self.tf1.get())
            bf2=str(self.bf2.get())
            tf2=str(self.tf2.get())
        else:
            #reverse the flanges
            bf2=str(self.bf1.get())
            tf2=str(self.tf1.get())
            bf1=str(self.bf2.get())
            tf1=str(self.tf2.get())

        


        # if M1=='0':
        #     tmp=hw1
        #     hw1=hw2
        #     hw2=tmp
        #     tmp=M1
        #     M1=str(-1*float(M2))
        #     M2=0
        #     #print("M1",M1,"M2",M2)

        
        the_min=min(float(M1),float(M2))/MMax
        the_max=max(float(M1),float(M2))/MMax

        psi=the_min/the_max

        #check the ranges of input values and warn the user if outside the range
        if not self.MakeNNRangeChecks(float(hw1),float(hw2),float(tw),float(bf1),float(tf1),float(bf2),float(tf2),float(L)):
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

        start = timer()
        Mcr=CalcMcr_NN(float(hw1),float(hw2),float(tw),float(bf1),float(tf1),float(bf2),float(tf2),psi,float(L),model,model_scaler,u)*10**-6
        end = timer()
        txt_result='alphacr= '+str(round(Mcr/MMax,3))
        txt_result+='\n\nMcr= '+str(round(Mcr,3))+' kN.m'
        txt_result+='\n\npsi='+str(psi)
        txt_result+='\n\ncalculation time was '+str(round(end-start,5))+' seconds'
        self.Results.configure(text='Results:\n\n'+txt_result, justify='left')

    def GButton_84_command_old(self):

        if not self.MakeInputChecks():
            return

        self.SaveFile()

        hw1=str(self.hw1.get())
        hw2=str(self.hw2.get())
        tw=str(self.tw.get())
        M1=float(str(self.M1.get()))
        M2=float(str(self.M2.get()))

        if float(hw1)>float(hw2):
            #we reverse the sides and the moments
            #this is to have always consistent results
            #the NN were trained ignoring this fact
            tmp=hw1
            hw1=hw2
            hw2=tmp
            tmp=M1
            M1=M2
            M2=tmp


        reverse_flanges=False
        if abs(M1)>=abs(M2):
            if M1<=0:
                #reverse_flanges=False
            #else:
                M1*=-1
                M2*=-1
                reverse_flanges=True
        else:
            #we need to reverse hw1 and hw2
            tmp=hw1
            hw1=hw2
            hw2=tmp
            if M2<=0:
                M1*=-1
                M2*=-1
                reverse_flanges=True
        L=str(self.L.get())

        if not reverse_flanges:
            bf1=str(self.bf1.get())
            tf1=str(self.tf1.get())
            bf2=str(self.bf2.get())
            tf2=str(self.tf2.get())
        else:
            #reverse the flanges
            bf2=str(self.bf1.get())
            tf2=str(self.tf1.get())
            bf1=str(self.bf2.get())
            tf1=str(self.tf2.get())

        MMax=max(abs(float(M1)),abs(float(M2)))


        # if M1=='0':
        #     tmp=hw1
        #     hw1=hw2
        #     hw2=tmp
        #     tmp=M1
        #     M1=str(-1*float(M2))
        #     M2=0
        #     #print("M1",M1,"M2",M2)

        
        the_min=min(float(M1),float(M2))/MMax
        the_max=max(float(M1),float(M2))/MMax

        psi=the_min/the_max

        #check the ranges of input values and warn the user if outside the range
        self.MakeNNRangeChecks(float(hw1),float(hw2),float(tw),float(bf1),float(tf1),float(bf2),float(tf2),float(L))


            
            
            

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
        self.Results.configure(text='Results:\n\n'+txt_result, justify='left')

        



if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)

    #this only works here, for any reason
    #bg_img = importimg('notation.png')
    
    #bg = tk.Label(root, image=bg_img)
    #bg.grid(column=0, row=0)

    model, model_scaler, u = LoadNN('./nn_data/9_128_16_1_lm_v1.nn')

    root.mainloop()
