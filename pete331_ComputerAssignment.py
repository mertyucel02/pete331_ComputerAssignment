# Computer Assignment - 20231
# PETE 331: Petroleum Production Engineering
# Department of Petroleum and Natural Gas Engineering
# Middle East Technical University
# Prepared by:
# Mert Yücel
# Yunus Emre Yalçın
# Onuralp Coşkun
# 05/01/2023

"""
MIT License

Copyright (c) 2024 Mert Yücel, Yunus Emre Yalçın, Onuralp Coşkun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Libraries used
import math
import numpy as np
from matplotlib import pyplot as plt
import statistics as stat
from tabulate import tabulate

# Input for student ID and the number of sections
id_num = input("Student ID: ")
section = int(input("# of Section: "))


# Function to convert Fahrenheit to Rankine
def F_to_R(temp_F):
    temp_R = temp_F + 460
    return temp_R


# Function to calculate pressure, temperature, and depth using SRK EoS
def AverageTZ_SRK_EoS(id_num, section):
    # Extract digits from student ID
    student_id = []
    for i in str(id_num):
        student_id.append(int(i))

    # Constant data derived from student ID
    q_sc = 3000 + (student_id[6] + 1) * 600  # Mscf / d
    gamma_g = .65 + ((student_id[4] + 1) / 50)
    length = 4000 + (student_id[5] + 1) * 500  # ft
    p_hf = 150 + (student_id[6] + 1) * 30  # psi
    theta = student_id[3] * 2  # degrees
    d_in = 2.44  # in.
    eps_d = .0041
    T_hf = 110  # Fahrenheit
    gradient_T = 13 / 1000  # Fahrenheit / ft

    R = 10.732  # (ft^3 * psi) / (Rankine * lb.mol)
    depth = np.linspace(0, length, section + 1)  # array([0, depth[1], ... , depth[section]])
    pressure = np.zeros(section + 1)  # array([0, (0 * 1), ... , (0 * section)])
    pressure[0] = p_hf  # array([p_hf, (0 * 1), ... , (0 * section)])
    temp = np.zeros(section + 1)  # array([0, (0 * 1), ... , (0 * section)])
    temp[0] = T_hf  # array([T_hf, (0 * 1), ... , (0 * section)])

    # Mole fractions of components of gas dry composition
    y_CH4 = (student_id[5] + 1) * 7 / 100
    y_C2H6 = (student_id[6] + 1) * 2 / 100
    y_C3H8 = 1 - y_C2H6 - y_CH4

    # Acentric factors of components of dry gas composition
    acentric_factor_CH4 = .011
    acentric_factor_C2H6 = .099
    acentric_factor_C3H8 = .152

    # Acentric factor of dry gas composition
    acentric_factor = y_CH4 * acentric_factor_CH4 + y_C2H6 * acentric_factor_C2H6 + y_C3H8 * acentric_factor_C3H8

    # Pseudo critical properties by the Standing’s (1977) correlation for dry gases using gas gravity
    p_pc = 677 + 15 * gamma_g - 37.5 * (gamma_g ** 2)
    T_pc = 168 + 325 * gamma_g - 12.5 * (gamma_g ** 2)

    # Coefficients of SRK EoS to calculate Z-Factor
    coeff_a = .42727 * (((R ** 2) * (T_pc ** 2)) / p_pc)  # (ft^6 * psi) / (lb.mol^2)
    coeff_b = .08664 * ((R * T_pc) / p_pc)  # ft^3 / lb.mol

    # Molecular weight of the natural gas
    MW_air = 28.966  # lb / lb.mol
    MW_a = gamma_g * MW_air  # lb / lb.mol

    # Main loop for pressure calculation using SRK EoS
    for i in range(section):
        p_estimated = pressure[i] + .0001  # psi
        data_p = [p_estimated, pressure[i]]  # psi
        p_av = stat.mean(data_p)  # (sum(data_p) / len(data_p))
        temp[i + 1] = temp[i] + length * gradient_T / section  # Fahrenheit
        data_T = [temp[i], temp[i + 1]]  # Fahrenheit
        T_av = stat.mean(data_T)  # (sum(data_T) / len(data_T))

        T_reduced = F_to_R(T_av) / F_to_R(T_pc)
        alpha = (1 + (.480 + 1.574 * acentric_factor - .176 * (acentric_factor ** 2)) * (1 - math.sqrt(T_reduced))) ** 2

        coeff_A = (alpha * coeff_a * p_av) / ((R * F_to_R(T_av)) ** 2)  # (ft^3 * psi) / (Rankine * lb.mol)
        coeff_B = (coeff_b * p_av) / (R * F_to_R(T_av))

        coeff_1 = 1
        coeff_2 = -1
        coeff_3 = coeff_A - coeff_B - coeff_B ** 2  # (ft^3 * psi) / (Rankine * lb.mol)
        coeff_4 = -1 * coeff_A * coeff_B  # (ft^3 * psi) / (Rankine * lb.mol)

        array_coeff = np.array([coeff_1, coeff_2, coeff_3, coeff_4],dtype=float)  # array([float(coeff_1), float(coeff_2), float(coeff_3), float(coeff_4)])
        coeff_poly = np.poly1d(np.polyder(array_coeff))  # float(array_coeff[0]) * x ^ (len(array_coeff) - 1) + float(array_coeff[1]) * x ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * x ^ (1) + float(array_coeff[len(array_coeff) - 1]) * x ^ (0)

        soln = np.roots(coeff_poly)
        '''
        array([soln[0, 0], soln[0, 1], soln[0, 2]],
              [soln[1, 0], soln[1, 1], soln[1, 2]],
              ...
              [soln[section, 0], soln[section, 1], soln[section, 2]])
              
        '''

        poly_val = np.polyval(array_coeff, soln)

        '''
        array([float(array_coeff[0]) * soln[0, 0] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[0, 0] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[0, 0] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[0, 0] ^ (0), float(array_coeff[0]) * soln[0, 1] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[0, 1] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[0, 1] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[0, 1] ^ (0), float(array_coeff[0]) * soln[0, 2] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[0, 2] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[0, 2] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[0, 2] ^ (0)],
              [float(array_coeff[0]) * soln[1, 0] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[1, 0] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[1, 0] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[1, 0] ^ (0), float(array_coeff[0]) * soln[1, 1] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[1, 1] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[1, 1] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[1, 1] ^ (0), float(array_coeff[0]) * soln[1, 2] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[1, 2] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[1, 2] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[1, 2] ^ (0)],
              ...
              [float(array_coeff[0]) * soln[section, 0] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[section, 0] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[section, 0] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[section, 0] ^ (0), float(array_coeff[0]) * soln[section, 1] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[section, 1] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[section, 1] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[section, 1] ^ (0), float(array_coeff[0]) * soln[section, 2] ^ (len(array_coeff) - 1) + float(array_coeff[1]) * soln[section, 2] ^ (len(array_coeff) - 2) + ... + float(array_coeff[len(array_coeff) - 2]) * soln[section, 2] ^ (1) + float(array_coeff[len(array_coeff) - 1]) * soln[section, 2] ^ (0)])
        '''

        is_real = np.isreal(poly_val)

        '''
        array([True, False, False],
              [True, False, False],
              ...
              [True, False, False])
        '''
        # since in Z-Factor solution by using SRK EoS, there is one real solution for each depth, and the real solution is the first element for each row.
        # Because of np.isreal, the imaginary solutions were removed.

        z_av = np.extract(is_real, soln)

        '''
        array([soln[0, 0]],
              [soln[1, 0]],
              ...
              [soln[section, 0]])
              
        '''

        # Reynold's Number by using Lee-Gonzalez and Eakin (1966) Correlation
        density_gas = MW_a * p_av / (F_to_R(T_av) * R * z_av)   # lb / ft^3
        constant_K = (9.4 + 0.02 * MW_a) * (F_to_R(T_av) ** 1.5) / (209 + 19 * MW_a + F_to_R(T_av)) * (10 ** (-4))
        constant_X = 3.5 + (986 / F_to_R(T_av)) + .01 * MW_a
        constant_Y = 2.4 - .2 * constant_X
        viscosity = constant_K * np.exp(constant_X * ((density_gas / 62.4) ** constant_Y))
        N_Re = 20.1 * q_sc * gamma_g / (d_in * viscosity)

        # A condition with respect to the type of fluid flow, laminar or turbulent
        # If the type of fluid flow is laminar:
        if np.all(N_Re < 2000):
            f_F = 16 / N_Re
        # If the type of fluid flow is turbulent:
        else:
            # Chen’s correlation is

            f_F = (-4 * np.log10(eps_d / 3.7065 - 5.0452 / N_Re * np.log10((eps_d ** 1.1098) / 2.8257 + (7.149 / N_Re) ** .8981))) ** (-2)

        f_M = f_F * 4
        '''
        fF = Fanning Friction Factor 
        fM = Moody’s Friction Factor
        '''
        s = .0375 * gamma_g * length * math.cos(math.radians(theta)) / (z_av * F_to_R(T_av) * section)
        e_s = np.exp(s)
        p_2 = np.sqrt(e_s * np.power(pressure[i], 2) + (6.67 * (10 ** (-4)) * (e_s - 1) * f_M * np.power((q_sc * z_av * F_to_R(T_av)), 2) / (np.power(d_in, 5) * np.cos(np.radians(theta)))))

        # Iterative loop for refining pressure estimate
        while np.any(np.abs(p_2 - p_estimated) > .001):
            p_estimated = p_2

            p_av = (p_estimated + p_2) / 2
            T_reduced = F_to_R(T_av) / F_to_R(T_pc)
            alpha = (1 + (.480 + 1.574 * acentric_factor - .176 * (acentric_factor ** 2)) * (
                        1 - math.sqrt(T_reduced))) ** 2

            coeff_A = (alpha * coeff_a * p_av) / ((R * F_to_R(T_av)) ** 2)
            coeff_B = (coeff_b * p_av) / (R * F_to_R(T_av))

            coeff_1 = 1
            coeff_2 = -1
            coeff_3 = coeff_A - coeff_B - coeff_B ** 2
            coeff_4 = -1 * coeff_A * coeff_B

            solns = np.roots(array_coeff)

            z_av = solns[np.isreal(solns)].real

            s = .0375 * gamma_g * length * math.cos(math.radians(theta)) / (z_av * F_to_R(T_av) * section)
            e_s = np.exp(s)
            p_2 = np.sqrt(e_s * np.power(pressure[i], 2) + (6.67 * (10 ** (-4)) * (e_s - 1) * f_M * np.power((q_sc * z_av * F_to_R(T_av)), 2) / (np.power(d_in, 5) * np.cos(np.radians(theta)))))

        pressure[i + 1] = p_2[0]
    return depth, pressure, temp


depth, pressure, temperature = AverageTZ_SRK_EoS(id_num, section)

# Populate the table data
my_data = []
for i in range(0, section + 1):
    my_data.append([depth[i], temperature[i], pressure[i]])

# Define column headers for the table
head = ["Length (ft)", "Temperature (Fahr.)", "Pressure (psia)"]

# Print the tabulated data
print(tabulate(my_data, headers=head, tablefmt="grid"))
print("Bottom Hole Flowing Pressure is " + str(pressure[section]) + " psia")

# Set up plot parameters
plt.rcParams["figure.figsize"] = [10, 5]
plt.rcParams["figure.autolayout"] = True

# Plot pressure versus depth curve
plt.plot(pressure, depth, linewidth=1, marker=".", mec="r", mfc="r")
plt.xlabel("Pressure ($psi$)")
plt.ylabel("Length ($ft$)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.title("The Plot of Length ($ft$) versus Pressure ($psi$)")

# Annotate the graph with pressure-depth values
for xy in zip(pressure, depth):
    plt.annotate('(%.2f psi, %.2f ft)' % xy, xy=xy)

# Display the graph
plt.show()
