'''
Play around with the lay-up values on line 52 to run through iterations on this one.  I'm just leaving it on the one
I found to be the last one needed but it's all based on initial guesses so you'll want to run through a few.  Use the
carpet plots to find your values for E_xx and G_xy and plug those in on line 41 & 42
'''

import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm  # Matrix multiplication
from math import sin
from math import cos
from math import tan

def main():
    # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy] in lb/in & in-lb/in
    stress_resultant = np.array([[5000], [0], [900], [0], [0], [0]])

    # Enter a desired ply orientation angles in degrees here:
    angle_in_degrees = [0,0,0,0,0,0,0,0,0,0,0,0,45,-45,90,90,90,-45,45,0,0,0,0,0,0,0,0,0,0,0,0]

    N = len(angle_in_degrees)   # Number of plies
    t_ply = 0.005               # Ply thickness in in
    h = N * t_ply               # Laminate thickness in inches

    # Number of at each angle
    n_0  = angle_in_degrees.count(0)
    n_45 = 2 * angle_in_degrees.count(45)  # Using symmetry to save on processing resources
    n_90 = angle_in_degrees.count(90)

    # Actual percentages of each ply group
    n_0_percent = n_0/N
    n_45_percent = n_45/N
    n_90_percent = n_90/N

    # Allowable strains
    e_xxc = 0.003          # Allowable normal strain
    gamma_xyc = 0.005       # Allowable shear strain

    # Extrapolate E_xx and G_xy from the carpet plots and lay-up
    E_xx = 11 * (10**6)
    G_xy = 1.5 * (10**6)

    # Laminate strains
    e_xx = float(stress_resultant[0] / (E_xx * h))
    gamma_xy = float(stress_resultant[2] / (G_xy * h))

    # Factors of safety
    FS_axial = float(e_xxc / abs(e_xx))
    FS_shear = float(gamma_xyc / abs(gamma_xy))

    print("Total number of plies :" + format(N,'>6d'))

    print('\nNumber of plies in each ply group:')
    print('Plies at 0\N{DEGREE SIGN}:' + format(n_0,'>7d'))
    print('Plies at 45\N{DEGREE SIGN}:' + format(n_45,'>6d'))
    print('Plies at 90\N{DEGREE SIGN}:' + format(n_90,'>6d'))

    print('\nPercent of plies in each ply group:')
    print('Plies at 0\N{DEGREE SIGN}:' + format(n_0_percent,'>7.2f'))
    print('Plies at 45\N{DEGREE SIGN}:' + format(n_45_percent,'>6.2f'))
    print('Plies at 90\N{DEGREE SIGN}:' + format(n_90_percent,'>6.2f'))

    print('\ne_xx for this iteration: ' + format(e_xx,'>10.5f'))
    print('γ_xy for this iteration: ' + format(gamma_xy,'>10.5f'))

    print('\nAxial Factor of Safety:' + format(FS_axial,'>12.3f'))
    print('Shear Factor of Safety:' + format(FS_shear,'>12.3f'))


main()