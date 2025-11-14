#!/usr/bin/env python3
from rk_4 import runge_kutta_iv

import matplotlib.pyplot as plt
import numpy as np
from settings import *

def punto_6():
    # Usar K1 = 120 y K2 = 0.7 en settings.py
    time, position, velocity, acceleration = runge_kutta_iv(0.35622478902624444, t_max=60)
    # position_peaks, _ = find_peaks(position)
    peak = max(position)

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
    plt.savefig('rk4_punto_6.png', dpi=300, bbox_inches='tight')
    plt.show()                           # Show the figure.


if __name__ == "__main__":
    punto_6()
