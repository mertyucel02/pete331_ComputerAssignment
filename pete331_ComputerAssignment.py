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

import math
import numpy as np
from matplotlib import pyplot as plt
import statistics as stat
from tabulate import tabulate

id_num = input("Student ID: ")
section = int(input("# of Section: "))

def F_to_R(temp_F):
    temp_R = temp_F + 460
    return temp_R


def AverageTZ_SRK_EoS(id_num, section):

    student_id = []

    for i in str(id_num):
        student_id.append(int(i))
    q_sc = 3000 + (student_id[6] + 1) * 600
    gamma_g = .65 + ((student_id[4] + 1) / 50)
    length = 4000 + (student_id[5] + 1) * 500
    p_hf = 150 + (student_id[6] + 1) * 30
    theta = student_id[3] * 2
    d_in = 2.44
    eps_d = .0041
    T_hf = 110
    gradient_T = 13 / 1000

    R = 10.732
    depth = np.linspace(0, length, section + 1)
    pressure = np.zeros(section + 1)
    pressure[0] = p_hf
    temp = np.zeros(section + 1)
    temp[0] = T_hf



    y_CH4 = (student_id[5] + 1) * 7 / 100
    y_C2H6 = (student_id[6] + 1) * 2 / 100
    y_C3H8 = 1 - y_C2H6 - y_CH4

    acentric_factor_CH4 = .011
    acentric_factor_C2H6 = .099
    acentric_factor_C3H8 = .152

    acentric_factor = y_CH4 * acentric_factor_CH4 + y_C2H6 * acentric_factor_C2H6 + y_C3H8 * acentric_factor_C3H8

    p_pc = 677 + 15 * gamma_g - 37.5 * (gamma_g ** 2)
    T_pc = 168 + 325 * gamma_g - 12.5 * (gamma_g ** 2)

    coeff_a = .42727 * (((R ** 2) * (T_pc ** 2)) / p_pc)
    coeff_b = .08664 * ((R * T_pc) / p_pc)

    for i in range(section):
        p_estimated = pressure[i] + .00000000001
        data_p = [p_estimated, pressure[i]]
        p_av = stat.mean(data_p)
        temp[i + 1] = temp[i] + length * gradient_T / section
        data_T = [temp[i], temp[i + 1]]
        T_av = stat.mean(data_T)

        T_reduced = F_to_R(T_av) / F_to_R(T_pc)
        alpha = (1 + (.480 + 1.574 * acentric_factor - .176 * (acentric_factor ** 2)) * (1 - math.sqrt(T_reduced))) ** 2

        coeff_A = (alpha * coeff_a * p_av) / ((R * F_to_R(T_av)) ** 2)
        coeff_B = (coeff_b * p_av) / (R * F_to_R(T_av))

        coeff_1 = 1
        coeff_2 = -1
        coeff_3 = coeff_A - coeff_B - coeff_B ** 2
        coeff_4 = -1 * coeff_A * coeff_B

        array_coeff = np.array([coeff_1, coeff_2, coeff_3, coeff_4], dtype=float)
        coeff_poly = np.poly1d(array_coeff)


        soln = np.roots(coeff_poly)


        z_av = np.extract(np.isreal(np.polyval(array_coeff, soln)), soln)
        if np.all(d_in <= 4.277):
            f_M = .01750 / d_in ** .224

        else:
            f_M = .01603 / d_in ** .164

        s = .0375 * gamma_g * length * math.cos(math.radians(theta)) / (z_av * F_to_R(T_av) * section)
        e_s = np.exp(s)
        p_2 = np.sqrt(e_s * np.power(pressure[i], 2) + (6.67 * (10 ** (-4)) * (e_s - 1) * f_M * np.power((q_sc * z_av * F_to_R(T_av)), 2) / (np.power(d_in, 5) * np.cos(np.radians(theta)))))

        while np.any(np.abs(p_2 - p_estimated) > 0.1):
            p_estimated = p_2

            p_av = (p_estimated + p_2) / 2
            T_reduced = F_to_R(T_av) / F_to_R(T_pc)
            alpha = (1 + (.480 + 1.574 * acentric_factor - .176 * (acentric_factor ** 2)) * (1 - math.sqrt(T_reduced))) ** 2

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
my_data = []
for i in range(0, section + 1):
    my_data.append([depth[i], temperature[i], pressure[i]])

head = ["Length (ft)", "Temperature (Fahr.)", "Pressure (psia)"]
print(tabulate(my_data, headers= head, tablefmt="grid"))
print("Bottom Hole Flowing Pressure is " + str(pressure[section]) + " psia")
plt.rcParams["figure.figsize"] = [15, 7]
plt.rcParams["figure.autolayout"] = True
plt.plot(pressure, depth, linewidth = 1, marker = ".")
plt.xlabel("Pressure (psia)")
plt.ylabel("Length (ft)")
plt.gca().invert_yaxis()
plt.grid(True)
plt.title("The Graph of Length (ft) versus Pressure (psia)")
for xy in zip(pressure, depth):
    plt.annotate('(%.2f, %.2f)' % xy, xy=xy)
plt.show()
