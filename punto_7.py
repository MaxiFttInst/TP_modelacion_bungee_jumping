#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from settings import *

def runge_kutta_iv(t_target, h):
    steps = int(t_target / h)
    t = np.linspace(0, t_target, steps + 1)
    u = np.zeros(steps + 1)
    v = np.zeros(steps + 1)
    a = np.zeros(steps + 1)

    def acc(u_val, v_val):
        f_viscosa = -C1 * abs(v_val)**C2 * np.sign(v_val)

        if u_val > L0:
            stretch = u_val - L0
            f_elastica = -K1 * stretch**K2
        else:
            f_elastica = 0.0
        fuerza_total = M * G + f_elastica + f_viscosa
        return fuerza_total / M

    for n in range(steps):
        # k1
        k1u = h * v[n]
        k1v = h * acc(u[n], v[n])

        # k2
        k2u = h * (v[n] + 0.5 * k1v)
        k2v = h* acc(u[n] + 0.5 * k1u, v[n] + 0.5 * k1v)

        # k3
        k3u = h* (v[n] + 0.5 * k2v)
        k3v = h * acc(u[n] + 0.5 * k2u, v[n] + 0.5 * k2v)

        # k4
        k4u = h * (v[n] + k3v)
        k4v = h* acc(u[n] + k3u, v[n] + k3v)

        # Actualización
        u[n+1] = u[n] + (1/6) * (k1u + 2*k2u + 2*k3u + k4u)
        v[n+1] = v[n] + (1/6) * (k1v + 2*k2v + 2*k3v + k4v)
        a[n+1] = acc(u[n+1], v[n+1])

    return t, u, v, a

def punto_7():
    # Configurar en settings los valores provistos en el informe
    time, position, velocity, acceleration = runge_kutta_iv(60,h=0.01)
    peak = np.max(position)

    theorical_peak = L0 + (2*M*G)/(2*K1) + np.sqrt((2*M*G*L0)/K1 + ((M*G)/K1)**2)
    relative_error  = abs(peak-theorical_peak)/theorical_peak
    print(f"L0: {L0}")
    print(f"H: {H}")
    print(f"M: {M}")
    print(f"G: {G}")
    print(f"K1: {K1}")
    print(f"máximo teórico: {theorical_peak}")
    print(f"máximo práctico: {peak}")
    print(f"Erorr relativo: {relative_error}")

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(time, position, 'b', label='Posición [m]')
    # axs[0].scatter(time[position_peaks], position[position_peaks],
    #                color='green', marker='^', label='Picos')

    axs[0].axhline(y=135, color='green', linestyle='--', linewidth=2,
            label=f'Caída al 90%', alpha=0.7)
    axs[0].axhline(y=150, color='red', linestyle='--', linewidth=2,
            label=f'Caída al 100%', alpha=0.7)
    axs[0].legend(fontsize=9)
    axs[0].set_ylabel("Posición [m]")

    axs[1].plot(time, list(map(lambda x: x * 3.6, velocity)), 'r', label='Velocidad [Km/h]')
    axs[1].set_ylabel("Velocidad [km/h]")

    axs[2].plot(time, list(map(lambda x: x / G,acceleration)), 'g', label='Aceleración [g]')
    axs[2].axhline(y=-2.5, color='red', linestyle='--', linewidth=2,
            label=f'Límite de fuerza G (-2.5G)', alpha=0.7)
    axs[2].set_ylabel("Aceleración [g]")
    axs[2].legend(fontsize=9)
    axs[2].set_xlabel("Tiempo [s]")
    plt.savefig('rk4_punto_7.png', dpi=300, bbox_inches='tight')
    plt.show()                           # Show the figure.

if __name__ == "__main__":
    punto_7()
