import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm  # Matrix multiplication
from math import sin
from math import cos
from math import tan

def main():  # Plain stress approximation
    # Independent material properties for AS/3501 graphite epoxy in US units
    E11  = 20.01 * (10**6) # psi
    E22  = 1.3   * (10**6) # psi
    V12  = 0.3             # unit-less
    V21 = (V12*E22)/E11    # unit-less
    G12  = 1.03  * (10**6) # psi

    # Typical strengths of AS/3501 graphite epoxy in US units
    SLt  = 209.9 * (10**3)  # psi
    SLc  = 209.9 * (10**3)  # psi
    STt  = 7.50  * (10**3)  # psi
    STc  = 29.9  * (10**3)  # psi
    SLTs = 13.5  * (10**3)  # psi

    '''
    # THESE WERE USED FOR CALCULATION VALIDATION BASED ON EXAMPLE pg253 2 FROM THE FILES SECTION ON CANVAS
    # Independent material properties for T300/5208 graphite epoxy in US units
    E11  = 26.25 * (10**6)  # psi
    E22  = 1.49  * (10**6)  # psi
    V12  = 0.28             # unit-less
    V21 = (V12*E22)/E11     # unit-less
    G12  = 1.04  * (10**6)  # psi

    # Typical strengths of T300/5208 graphite epoxy in US units
    SLt  = 217.5 * (10**3)  # psi
    SLc  = 217.5 * (10**3)  # psi
    STt  = 5.80  * (10**3)  # psi
    STc  = 35.7  * (10**3)  # psi
    SLTs = 9.86  * (10**3)  # psi
    '''

    # Tsai-Wu Coefficients
    F11 = 1 / (SLt*SLc)
    F22 = 1 / (STt*STc)
    F12 = (-1/2) * math.sqrt(F11* F22)
    F66 = 1 / (SLTs**2)
    F1  = (1/SLt) - (1/SLc)
    F2  = (1/STt) - (1/STc)

    # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy] in lb/in & in-lb/in
    stress_resultant = np.array([[4000], [0], [800], [0], [0], [0]])

    # # Used to verify code based on Example 253 2
    # stress_resultant = np.array([[8000], [0], [1300], [0], [0], [0]])

    # Enter a desired ply orientation angles in degrees here:
    angle_in_degrees = [0,0,0,0,45,-45,90,90,-45,45,0,0,0,0]

    N = len(angle_in_degrees)   # Number of plies
    t_ply = 0.006               # Ply thickness in in
    h = N * t_ply               # Laminate thickness in inches

    # Number of at each angle
    n_0  = angle_in_degrees.count(0)
    n_45 = 2 * angle_in_degrees.count(45)  # Using symmetry to save on processing resources
    n_90 = angle_in_degrees.count(90)

    # Allowable strains
    e_xxc = 0.004          # Allowable normal strain
    gamma_xyc = 0.005       # Allowable shear strain

    # Distance from laminate mid-plane to out surfaces of plies)
    z0 = -h/2
    z = [0] * (N)
    for i in range(N):
        z[i] = (-h / 2) + ((i+1) * t_ply)

    # Distance from laminate mid-plane to mid-planes of plies
    z_mid_plane = [0] * N
    for i in range(N):
        z_mid_plane[i] = (-h / 2) - (t_ply/2) + ((i+1) * t_ply)

    # Ply orientation angle translated to radians to simplify equations below
    angle = [0] * N
    for i in range(N):
        angle[i] = math.radians(angle_in_degrees[i])

    # Stress Transformation (Global to Local), pg 112
    T = [0] * N
    for i in range(N):
        T[i] = np.array([[cos(angle[i])**2, sin(angle[i])**2, 2*sin(angle[i])*cos(angle[i])],
                         [sin(angle[i])**2, cos(angle[i])**2, -2*sin(angle[i])*cos(angle[i])],
                         [-sin(angle[i])*cos(angle[i]), sin(angle[i])*cos(angle[i]), cos(angle[i])**2-sin(angle[i])**2]])

    # Strain Transformation (Global-to-Local), pg 113
    T_hat = [0] * N
    for i in range(N):
        T_hat[i] = np.array([[cos(angle[i])**2, sin(angle[i])**2, sin(angle[i])*cos(angle[i])],
                             [sin(angle[i])**2, cos(angle[i])**2, -sin(angle[i])*cos(angle[i])],
                             [-2*sin(angle[i])*cos(angle[i]), 2*sin(angle[i])*cos(angle[i]), cos(angle[i])**2-sin(angle[i])**2]])

    # The local/lamina compliance matrix, pg 110
    S11 = 1/E11
    S12 = -V21/E22
    S21 = -V12/E11
    S22 = 1/E22
    S33 = 1/G12
    S = np.array([[S11, S12, 0], [S21, S22, 0], [0, 0, S33]])

    # The local/lamina stiffness matrix, pg 107
    Q_array = lg.inv(S)  # The inverse of the S matrix

    # The global/laminate stiffness and compliance matrices
    Q_bar_array = [0] * N
    for i in range(N):
        Q_bar_array[i] = mm(lg.inv(T[i]), mm(Q_array, T_hat[i]))  # The global/laminate stiffness matrix, pg 114

    A_array = [[0]*3]*3
    for i in range(N):
        A_array += Q_bar_array[i] * t_ply

    # Transforming numpy array into lists for ease of formatting
    Q = Q_array.tolist()
    Q_bar = [0] * N
    for i in range(N):
        Q_bar[i] = Q_bar_array[i].tolist()
    A = A_array.tolist()


    # Calculating E_xx and G_xy
    E_xx = (A[0][0]/h) * (1 - ((A[0][1]**2)/(A[0][0]*A[1][1])))
    G_xy = A[2][2] / h

    # Calculating Nxx & Nxy
    N_xx = E_xx * e_xxc * h
    N_xy = G_xy * gamma_xyc * h

    # Factors of safety
    FS_axial = float(N_xx / stress_resultant[0])
    FS_shear = float(N_xy / stress_resultant[2])

    print("Total number of plies :" + format(N,'>6d'))
    print('\nNumber of plies in each ply group:')
    print('Plies at 0\N{DEGREE SIGN}:' + format(n_0,'>7d'))
    print('Plies at 45\N{DEGREE SIGN}:' + format(n_45,'>6d'))
    print('Plies at 90\N{DEGREE SIGN}:' + format(n_90,'>6d'))
    print('\nFactor of Safety for N_xx:' + format(FS_axial,'>6.3f'))
    print('Factor of Safety for N_xy:' + format(FS_shear,'>6.3f'))


main()