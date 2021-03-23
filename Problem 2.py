import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm  # Matrix multiplication
from math import sin
from math import cos
from math import tan


def main():  # Plain stress approximation
    # Independent material properties for AS/3501 graphite epoxy in SI units
    # E11 = 138  *(10**9)   # GPa
    # E22 = 8.96 *(10**9)   # GPa
    # V12 = 0.3             # unit-less
    # G12 = 7.1  *(10**9)   # GPa

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
    THESE WERE USED FOR CALCULATION VALIDATION BASED ON EXAMPLE pg226 FROM THE FILES SECTION ON CANVAS
    # Independent material properties for T300/5208 graphite epoxy in US units
    # E11  = 26.25 * (10**6)  # psi
    # E22  = 1.49  * (10**6)  # psi
    # V12  = 0.28             # unit-less
    # V21 = (V12*E22)/E11     # unit-less
    # G12  = 1.04  * (10**6)  # psi
    #
    # # Typical strengths of T300/5208 graphite epoxy in US units
    # SLt  = 217.5 * (10**3)  # psi
    # SLc  = 217.5 * (10**3)  # psi
    # STt  = 5.80  * (10**3)  # psi
    # STc  = 35.7  * (10**3)  # psi
    # SLTs = 9.86  * (10**3)  # psi
    '''

    # Tsai-Wu Coefficients
    F11 = 1 / (SLt*SLc)
    F22 = 1 / (STt*STc)
    F12 = (-1/2) * math.sqrt(F11* F22)
    F66 = 1 / (SLTs**2)
    F1  = (1/SLt) - (1/SLc)
    F2  = (1/STt) - (1/STc)

    # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy] in N/m & N-m/m
    stress_resultant = np.array([[100], [0], [0], [0], [0], [-10]])

    N = 8                   # number of plies
    t_ply = 0.005           # ply thickness in m
    t_LAM = t_ply * N       # laminate thickness in m

    # Distance from laminate mid-plane to out surfaces of plies)
    z0 = -t_LAM/2
    z = [0] * (N)
    for i in range(N):
        z[i] = (-t_LAM / 2) + ((i+1) * t_ply)

    # Distance from laminate mid-plane to mid-planes of plies
    z_mid_plane = [0] * N
    for i in range(N):
        z_mid_plane[i] = (-t_LAM / 2) - (t_ply/2) + ((i+1) * t_ply)

    # Enter a desired ply orientation angle in degrees here:
    angle_in_degrees = [0, 45, 90, -45, -45, 90, 45, 0]

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
    ''' # Calculated manually, not necessary if S matrix is known, pg 110
    Q11 = E11/(1-V12*V21)
    Q12 = (V21*E11)/(1-V12*V21)
    Q21 = (V12*E22)/(1-V12*V21)
    Q22 = E22/(1-V12*V21)
    Q = np.array([[Q11, Q12, 0], [Q21, Q22, 0], [0, 0, G12]])
    '''

    # The global/laminate stiffness and compliance matrices
    Q_bar_array = [0] * N
    for i in range(N):
        Q_bar_array[i] = mm(lg.inv(T[i]), mm(Q_array, T_hat[i]))  # The global/laminate stiffness matrix, pg 114

    A_array = [[0]*3]*3
    for i in range(N):
        A_array += Q_bar_array[i] * t_ply

    B_array = [[0]*3]*3
    for i in range(N):
        B_array += (1/2) * (Q_bar_array[i] * ((z[i]**2) - ((z[i] - t_ply)**2)))

    D_array = [[0] * 3] * 3
    for i in range(N):
        D_array += (1/3) * (Q_bar_array[i] * ((z[i] ** 3) - ((z[i] - t_ply) ** 3)))

    ABD_array = np.array([[A_array[0][0], A_array[0][1], A_array[0][2], B_array[0][0], B_array[0][1], B_array[0][2]],
                          [A_array[1][0], A_array[1][1], A_array[1][2], B_array[1][0], B_array[1][1], B_array[1][2]],
                          [A_array[2][0], A_array[2][1], A_array[2][2], B_array[2][0], B_array[2][1], B_array[2][2]],
                          [B_array[0][0], B_array[0][1], B_array[0][2], D_array[0][0], D_array[0][1], D_array[0][2]],
                          [B_array[1][0], B_array[1][1], B_array[1][2], D_array[1][0], D_array[1][1], D_array[1][2]],
                          [B_array[2][0], B_array[2][1], B_array[2][2], D_array[2][0], D_array[2][1], D_array[2][2]]])

    ABD_inverse_array = lg.inv(ABD_array)

    # Calculating the mid-plane strains and curvatures
    mid_plane_strains_and_curvatures_array = mm(lg.inv(ABD_array), stress_resultant)

    # Transforming numpy array into lists for ease of formatting
    Q = Q_array.tolist()
    Q_bar = [0] * N
    for i in range(N):
        Q_bar[i] = Q_bar_array[i].tolist()
    A = A_array.tolist()
    B = B_array.tolist()
    D = D_array.tolist()
    ABD_inverse = ABD_inverse_array.tolist()
    mid_plane_strains_and_curvatures = mid_plane_strains_and_curvatures_array.tolist()

    # Parsing the Mid-plane strains and curvatures apart
    mid_plane_strains = np.array([[mid_plane_strains_and_curvatures[0][0]], [mid_plane_strains_and_curvatures[1][0]], [mid_plane_strains_and_curvatures[2][0]]])
    curvatures = np.array([[mid_plane_strains_and_curvatures[3][0]], [mid_plane_strains_and_curvatures[4][0]], [mid_plane_strains_and_curvatures[5][0]]])

    # Global Strains at mid-plane of each ply
    global_strains = [[[0]]*3]*N
    for i in range(N):
        global_strains[i] = mid_plane_strains + z_mid_plane[i]*curvatures

    # Global Stresses at mid-plane of each ply
    global_stresses = [[[0]]*3]*N
    for i in range(N):
        global_stresses[i] = mm(Q_bar[i], global_strains[i])

    # Local strains
    local_strains = [[[0]]*3]*N
    for i in range(N):
        local_strains[i] = mm(T_hat[i], global_strains[i])

    # Local stresses
    local_stresses = [[[0]]*3]*N
    for i in range(N):
        local_stresses[i] = mm(Q, local_strains[i])

    # Strength Ratios for Max Stress Failure Theory
    R_sig_11 = [0]*N
    for i in range(N):
        R_sig_11[i] = SLt / math.fabs(local_stresses[i][0])

    R_sig_22 = [0] * N
    for i in range(N):
        R_sig_22[i] = STt / math.fabs(local_stresses[i][1])

    R_tau_12 = [0] * N
    for i in range(N):
        R_tau_12[i] = SLTs/ math.fabs(local_stresses[i][2])

    R_MS = [0]*N
    for i in range(N):
        R_MS[i] = min(R_sig_11[i], R_sig_22[i], R_tau_12[i])

    # Max stress critical loads
    N_MS_xxc = [0]*N
    for i in range(N):
        N_MS_xxc[i] = R_MS[i] * stress_resultant[0]

    M_MS_xxc = [0] * N
    for i in range(N):
        M_MS_xxc[i] = R_MS[i] * stress_resultant[5]

    # Define Tsai-Wu quadratic function coefficients (aR^2 + bR + cc = 0)
    a = [0]*N
    for i in range(N):
        a[i] = (F11*(local_stresses[i][0]**2)) + (2*F12*local_stresses[i][0]*local_stresses[i][1]) + (F22*(local_stresses[i][1]**2)) + (F66*(local_stresses[i][2]**2))

    b = [0]*N
    for i in range(N):
        b[i] = (F1*local_stresses[i][0]) + (F2*local_stresses[i][1])

    cc = [-1]*N

    # Strength Ratios for Tsai-Wu Criteria
    R_1_array = [0]*N
    for i in range(N):
        R_1_array[i] = (-b[i]+math.sqrt((b[i]**2)-4*a[i]*cc[i])) / (2*a[i])

    R_2 = [0] * N
    for i in range(N):
        R_2[i] = (-b[i] - math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i])

    R_1 = [0]*N
    for i in range(N):
        R_1[i] = R_1_array[i].tolist()
    R_TW = min(R_1)

    # Tsai-Wu critical loads
    N_TW_xxc = R_TW * stress_resultant[0]
    M_TW_xxc = R_TW * stress_resultant[5]

    # Printing the Strength Ratio for Max Stress Failure
    print("This is the Strength Ratio for first ply failure under Max Stress Failure Criterion:")
    print("R\N{LATIN SUBSCRIPT SMALL LETTER M}\N{LATIN SUBSCRIPT SMALL LETTER S} = " + str(round(min(R_MS, key=abs), 3)))
    print("# of ply that fails first: " + str(R_MS.index(min(R_MS)) + 1))

    # Printing the Strength Ratio for Tsai-Wu Failure
    print("\nThis is the Strength Ratio for the first ply failure under Tsai-Wu Failure Criterion:")
    print("R_TW = " + str(np.round(abs(min(R_TW, key=abs)), 3)))
    print("# of ply that fails first: " + str(R_1.index(min(R_1)) + 1))

    # Printing the Critical loads
    print("\nThese are the Critical Loads for first ply failure:")
    print("N_xx = " + str(np.round(abs(min(N_MS_xxc[0], N_TW_xxc[0])), 2)))
    print("M_xy = " + str(np.round(abs(min(M_MS_xxc[0], M_TW_xxc[0], key=abs)), 2)))


main()