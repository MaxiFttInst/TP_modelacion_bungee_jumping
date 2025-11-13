#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from settings import *

def grafico_error_vs_h():
    h_values, error_values = h_optimo()

    error_objetivo = 0.1
    plt.figure(figsize=(10, 6))
    p, logC = np.polyfit(np.log(h_values), np.log(error_values), 1)
    print(f"Orden experimental ≈ {p:.2f}")
    plt.loglog(h_values, error_values, 'o-', linewidth=2, markersize=6,
            label='Error numérico', color='blue')

    # Línea de referencia con pendiente 1 (orden 1)
    h_ref = np.array([h_values[0], h_values[-1]])
    error_ref = error_values[0] * (h_ref / h_values[0])**4
    plt.loglog(h_ref, error_ref, '--', linewidth=2, color='red',
            label='Pendiente 4 (orden 4)', alpha=0.7)

    # Línea horizontal en 0.1% (objetivo)
    plt.axhline(y=error_objetivo, color='green', linestyle='--', linewidth=2,
                label=f'Error objetivo (0.1%)', alpha=0.7)
    
    plt.xlabel('Paso temporal h [s]', fontsize=12)
    plt.ylabel('Error relativo [%]', fontsize=12)
    plt.title('Comprobación experimental del orden del Método de Runge-Kutta', fontsize=14)
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('rk_orden.png', dpi=300, bbox_inches='tight')
    plt.show()

def h_optimo():
    h_values = np.logspace(-5, 1, num = 30)  # 10 valores entre 0.1 y 1.0
    error_values = []
    for h in h_values:
        time, position, velocity, acceleration = runge_kutta_iv(8, h)
        max_y = np.max(position)
        theorical_peak = L0 + (2*M*G)/(2*K1) + np.sqrt((2*M*G*L0)/K1 + ((M*G)/K1)**2)
        error = 100 * abs(max_y - theorical_peak) / theorical_peak
        error_values.append(error)
    return h_values, error_values
def runge_kutta_iv(t_target, h):
    steps = int(t_target / h)
    t = np.linspace(0, t_target, steps + 1)
    u = np.zeros(steps + 1)
    v = np.zeros(steps + 1)
    a = np.zeros(steps + 1)

    def acc(u_val):
        if u_val > L0:
            stretch = u_val - L0
            return G - (K1 * stretch**K2) / M
        else:
            return G

    for n in range(steps):
        # k1
        k1u = h * v[n]
        k1v = h * acc(u[n])

        # k2
        k2u = h * (v[n] + 0.5 * k1v)
        k2v = h* acc(u[n] + 0.5 * k1u)

        # k3
        k3u = h* (v[n] + 0.5 * k2v)
        k3v = h * acc(u[n] + 0.5 * k2u)

        # k4
        k4u = h * (v[n] + k3v)
        k4v = h* acc(u[n] + k3u)

        # Actualización
        u[n+1] = u[n] + (1/6) * (k1u + 2*k2u + 2*k3u + k4u)
        v[n+1] = v[n] + (1/6) * (k1v + 2*k2v + 2*k3v + k4v)
        a[n+1] = acc(u[n+1])

    return t, u, v, a

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
    time, position, velocity, acceleration = runge_kutta_iv(6.7,h=0.5455594781168515)
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

    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(time, position, 'b', label='Posición [m]')
    # axs[0].scatter(time[position_peaks], position[position_peaks],
    #                color='green', marker='^', label='Picos')

    axs[0].axhline(y=theorical_peak, color='green', linestyle='--', linewidth=2,
            label=f'Pico teórico ({theorical_peak}%)', alpha=0.7)
    axs[0].legend(fontsize=9)
    axs[0].set_ylabel("Posición [m]")

    axs[1].plot(time, velocity, 'r--', label='Velocidad [Km/h]')
    axs[1].set_ylabel("Velocidad [m/s]")

    # axs[2].plot(time, list(map(lambda x: x / G,acceleration)), 'g', label='Aceleración [g]')
    # axs[2].set_ylabel("Aceleración [g]")
    axs[1].set_xlabel("Tiempo [s]")
    plt.savefig('rk4_trayectoria.png', dpi=300, bbox_inches='tight')
    plt.show()                           # Show the figure.


if __name__ == "__main__":
    main()
    h_values, error_values =  h_optimo()
    error_values = np.array(error_values)
    mask = error_values <= 0.1
    h_filtered = h_values[mask]
    error_filtered = error_values[mask]

    # Podés devolver ambos (filtrados y originales si querés)
    print(h_filtered[-1], error_filtered[-1])

    # grafico_error_vs_h()
