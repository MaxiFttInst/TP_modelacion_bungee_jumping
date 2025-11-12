#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from settings import *

def h_optimo():
    relative_error = 1
    h = 3
    peaks = 1
    while relative_error > 0.01 or peaks != 4:
        time, position, velocity, acceleration = runge_kutta_iv(50, h)
        position_peaks, _ = find_peaks(position)
        peak = max(list(position[position_peaks]))
        peaks = len(list(position[position_peaks]))

        theorical_peak = L0 + (2*M*G)/(2*K1) + np.sqrt((2*M*G*L0)/K1 + ((M*G)/K1)**2)
        relative_error  = abs(peak-theorical_peak)/theorical_peak
        print(f"h: {h}")
        print(f"L0: {L0}")
        print(f"H: {H}")
        print(f"M: {M}")
        print(f"G: {G}")
        print(f"K1: {K1}")
        print(f"Cantidad de picos: {peaks}")
        print(f"máximo teórico: {theorical_peak}")
        print(f"máximo práctico: {peak}")
        print(f"Erorr relativo: {relative_error}")
        h -= 0.0001
    return h

def runge_kutta_iv(t_target, h):
    steps = int(t_target/h)

    t = [0.0 for _ in range(steps + 1)]
    u = [0.0 for _ in range(steps + 1)]
    v = [0.0 for _ in range(steps + 1)]
    v_d = [0.0 for _ in range(steps + 1)]

    q1u = [0.0 for _ in range(steps + 1)]
    q2u = [0.0 for _ in range(steps + 1)]
    q3u = [0.0 for _ in range(steps + 1)]
    q4u = [0.0 for _ in range(steps + 1)]

    q1v = [0.0 for _ in range(steps + 1)]
    q2v = [0.0 for _ in range(steps + 1)]
    q3v = [0.0 for _ in range(steps + 1)]
    q4v = [0.0 for _ in range(steps + 1)]

    q1u[0] = h * v[0]
    q2u[0] = h * (v[0] + q1v[0]/2)
    q3u[0] = h * (v[0] + q2v[0]/2)
    q4u[0] = h * (v[0] + q3v[0])

    q1v[0] = h * G
    q2v[0] = h * G
    q3v[0] = h * G
    q4v[0] = h * G

    a = G
    u[1] = u[0] + (q1u[0] + 2*q2u[0] + 2*q3u[0] + q4u[0])/6
    
    v[1] = v[0] + (q1v[0] + 2*q2v[0] + 2*q3v[0] + q4v[0])/6

    v_d[1] = a
    t[1] = h

    for n in range(1,steps):
        if u[n] > L0:
            q1u[n] = h * v[n]
            stretch = max(u[n] - L0, 0)
            q1v[n] = h * (G - (K1*stretch**K2)/M)

            q2u[n] = h * (v[n] + q1v[n]/2)

            stretch = max(u[n] + q1u[n]/2 - L0, 0)
            q2v[n] = h * (G - (K1*stretch**K2)/M)

            q3u[n] = h * (v[n] + q2v[n]/2)

            stretch = max(u[n] + q2u[n]/2 - L0, 0)
            q3v[n] = h * (G - (K1*stretch**K2)/M)

            q4u[n] = h * (v[n] + q3v[n])

            stretch = max(u[n] + q3u[n] - L0, 0)
            q4v[n] = h * (G - (K1*stretch**K2)/M)
            a = G - (K1 * (u[n] - L0)**K2) / M
        else:
            q1v[n] = h * G
            q2v[n] = h * G
            q3v[n] = h * G
            q4v[n] = h * G
            q1u[n] = h * v[n]
            q2u[n] = h * (v[n] + q1v[n]/2)
            q3u[n] = h * (v[n] + q2v[n]/2)
            q4u[n] = h * (v[n] + q3v[n])
            a = G

        u[n + 1] = u[n] + (q1u[n] + 2*q2u[n] + 2*q3u[n] + q4u[n])/6
        v[n + 1] = v[n] + (q1v[n] + 2*q2v[n] + 2*q3v[n] + q4v[n])/6

        v_d[n] = a
        t[n + 1] = t[n] + h

    y = u # posición
    a = v_d # aceleración
    return np.array(t), np.array(y), np.array(v), np.array(a)

            
def main():
    ## TODAS LAS METRICAS USADAS EN METROS
    # time, position, velocity, acceleration = runge_kutta_iv(50)
    # ax.plot(time, position, 'b', label='Position')
    # ax.plot(time, velocity, 'r--', label='Velocity')
    # ax.plot(time, acceleration, 'g', label='Acceleration * 10^3')
    # ax.legend(['Posición', 'Velocidad', 'Aceleración * 10^3'])
    # plt.ylabel('metros')
    # plt.show()                           # Show the figure.
    #
    time, position, velocity, acceleration = runge_kutta_iv(50,h=0.6833999999980349)
    position_peaks, _ = find_peaks(position)
    peak = max(list(position[position_peaks]))
    peaks = len(list(position[position_peaks]))

    theorical_peak = L0 + (2*M*G)/(2*K1) + np.sqrt((2*M*G*L0)/K1 + ((M*G)/K1)**2)
    relative_error  = abs(peak-theorical_peak)/theorical_peak
    print(f"L0: {L0}")
    print(f"H: {H}")
    print(f"M: {M}")
    print(f"G: {G}")
    print(f"K1: {K1}")
    print(f"Cantidad de picos: {peaks}")
    print(f"máximo teórico: {theorical_peak}")
    print(f"máximo práctico: {peak}")
    print(f"Erorr relativo: {relative_error}")

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(time, position, 'b', label='Posición [m]')
    axs[0].scatter(time[position_peaks], position[position_peaks],
                   color='green', marker='^', label='Picos')
    axs[0].set_ylabel("Posición [m]")

    axs[1].plot(time, list(map(lambda x: (x * 3600)/1000,velocity)), 'r--', label='Velocidad [Km/h]')
    axs[1].set_ylabel("Velocidad [Km/h]")

    axs[2].plot(time, list(map(lambda x: x / G,acceleration)), 'g', label='Aceleración [g]')
    axs[2].set_ylabel("Aceleración [g]")
    axs[2].set_xlabel("Tiempo [s]")

    plt.show()                           # Show the figure.


if __name__ == "__main__":
    main()
