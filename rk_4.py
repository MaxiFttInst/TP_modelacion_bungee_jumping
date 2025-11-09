#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

NP = 111687
H = 150
G = 9.82
L0 = (0.1 / 10000 * (NP - 100000) + 0.25) * H 
K1 = 10 / 10000 * (NP - 100000) + 40
K2 = 1
M = 40 / 10000 * (NP - 100000) + 50 

def runge_kutta_iv(t_target):
    h = 0.01 
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

            q1v[n] = h * (G - (K1*pow(u[n] - L0, K2))/M)

            q2u[n] = h * (v[n] + q1v[n]/2)

            q2v[n] = h * (G - (K1*pow(u[n] + q1u[n]/2 - L0, K2))/M)

            q3u[n] = h * (v[n] + q2v[n]/2)

            q3v[n] = h * (G - (K1*pow(u[n] + q2u[n]/2 - L0, K2))/M)

            q4u[n] = h * (v[n] + q3v[n])

            q4v[n] = h * (G - (K1*pow(u[n] + q3u[n] - L0, K2))/M)
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

    return t, u, v, v_d

            
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
    time, position, velocity, acceleration = runge_kutta_iv(50)
    print(f"Long soga: {L0}")

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(time, position, 'b', label='Posición [m]')
    axs[0].set_ylabel("Posición [m]")

    axs[1].plot(time, list(map(lambda x: (x * 3600)/1000,velocity)), 'r--', label='Velocidad [Km/h]')
    axs[1].set_ylabel("Velocidad [Km/h]")

    axs[2].plot(time, list(map(lambda x: x / G,acceleration)), 'g', label='Aceleración [g]')
    axs[2].set_ylabel("Aceleración [g]")
    axs[2].set_xlabel("Time [s]")

    plt.show()                           # Show the figure.


if __name__ == "__main__":
    main()
