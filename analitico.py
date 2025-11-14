import numpy as np
from settings import *

# ============================================================================
# FUNCIÓN SOLUCIÓN ANALÍTICA
# ============================================================================

def solucion_analitica(t_array):
    """Calcula la solución analítica por tramos para múltiples oscilaciones."""
    y_array = np.zeros_like(t_array)
    v_array = np.zeros_like(t_array)
    a_array = np.zeros_like(t_array)
    
    # Tiempo en que alcanza L0 en caída libre
    t1 = np.sqrt(2 * L0 / G)
    v1 = G * t1
    
    # Parámetros del MAS
    omega = np.sqrt(K1 / M)
    A = v1 / omega
    B = M * G / K1
    
    # Período de oscilación
    T = 2 * np.pi / omega
    
    for i, t in enumerate(t_array):
        if t <= t1:
            # Fase 1: Caída libre inicial
            y_array[i] = 0.5 * G * t**2
            v_array[i] = G * t
            a_array[i] = G
        else:
            # Fase 2: Oscilación armónica continua
            tau = t - t1  # Tiempo desde que comenzó la oscilación
            
            y_array[i] = L0 + A * np.sin(omega * tau) + B * (1 - np.cos(omega * tau))
            v_array[i] = A * omega * np.cos(omega * tau) + B * omega * np.sin(omega * tau)
            a_array[i] = -A * omega**2 * np.sin(omega * tau) + B * omega**2 * np.cos(omega * tau)
    
    return y_array, v_array, a_array