import numpy as np

'''
What do I want to do????
In BEM code I want 2 modes
1) Uniform Inflow
2) Momentum theory With tip loss

Wait for design purposes I can use the 1st one but for detailed simulation I can use the second

We would also need a coordinate transformation, Keep in mind we have torque, we need gyroscopic moment 
which allows us to work angular acceleration.

I will add cyclic pitches also for the sake of it

So the functions are

def UniformInflow()

def NonUniformInflow()

What do I want each to return. 
Now I think we need to have 2 sets of outputs
Design mode and Simulator mode

Simulator mode would only be for advanced Code

updated set of functions

def UniformInflow()
    return CT,CL,CM,CX,CY,CQ

def NonUniformInflow()
    return CT,CL,CM,CX,CY,CQ


def Simulation()
    return C, lambda, beta

Cl, Cd, Cm of airfoil will have arrays, in which we vary alpha and Re, allowing us to import really precise data
Okay, here we go

I JUST REALISED THAT BEM REQUIRES US TO INPUT THE INFLOW WHAT DO I ASSUME THE INFLOW TO BE. BEMT WAS HOW WE GOT
AN INFLOW TO WORK WITH
'''

def UniformInflow(W, theta, n, N, Cl, Cd, c, twist, tap, R, rcut):
    return 0

def NonUniformInflow(W, th, n, N, Cl, Cd, c, twist, tap, R, rcut,psi,param):
    """"
    Param: 1 = Momentum theory inflow
           2 = Momentum theory + Prandtl Tip loss
    """

    psir = np.deg2rad(psi)
    r_v = np.linspace(rcut,R,n)
    theta = np.zeros(n,N)
    for i in range(n):
        for j in range(N):
            theta = (th[0] + th[1]*np.sin(psir+j*2*np.pi/N) + th[2]*np.cos(psir
                    +j*2*np.pi/N) + twist * (r_v(i)-rcut)/(R-rcut))

    Vv = np.zeros(n,N)

    if param == 1:
        for i in range(n):
            for j in range(N):
                Vv[i][j] = W*R*

        return 1
    elif param == 2:
        return 2
    else:
        return 0


    return 0
