import numpy as np
import matplotlib.pyplot as plt
from euler import euler_bungee
from rk4 import rk4_bungee
from analitico import solucion_analitica
from settings import *

# ============================================================================
# PARÁMETROS
# ============================================================================

NP = 111687

print("=" * 70)
print("COMPARACIÓN DE MÉTODOS NUMÉRICOS - 4 CAÍDAS SUCESIVAS")
print("=" * 70)
print(f"\nParámetros (NP = {NP}):")
print(f"  L0 = {L0:.2f} M")
print(f"  M = {M:.2f} kg")
print(f"  K1 = {K1:.2f} N/M")
print()

# Pasos óptimos encontrados en los ítems anteriores
h_euler = 0.002929
h_rk4 = 1.0

print(f"Pasos temporales utilizados:")
print(f"  Euler: h = {h_euler:.6f} s")
print(f"  RK4:   h = {h_rk4:.6f} s")
print()

# ============================================================================
# VALOR ANALÍTICO DEL PUNTO MÁS BAJO
# ============================================================================

mg_k1 = M * G / K1
y_max_analitico = L0 + mg_k1 + np.sqrt(mg_k1**2 + 2*M*G*L0/K1)

# ============================================================================
# SIMULACIONES
# ============================================================================

print("Ejecutando simulaciones...")

# Euler (sin detectar primer máximo)
t_euler, y_euler, v_euler, _ = euler_bungee(h_euler, t_max=30, detectar_primer_maximo=False)

# RK4 (sin detectar primer máximo)
t_rk4, y_rk4, v_rk4, _ = rk4_bungee(h_rk4, t_max=30, detectar_primer_maximo=False)

# Calcular aceleraciones
a_euler = np.array([G if y <= L0 else G - (K1/M)*(y - L0) for y in y_euler])
a_rk4 = np.array([G if y <= L0 else G - (K1/M)*(y - L0) for y in y_rk4])

# Solución analítica
t_analitico = np.linspace(0, min(t_euler[-1], t_rk4[-1]), 5000)
y_analitico, v_analitico, a_analitico = solucion_analitica(t_analitico)

print(f"Euler:     {len(t_euler)} puntos")
print(f"RK4:       {len(t_rk4)} puntos")
print(f"Analítico: {len(t_analitico)} puntos")
print()

# ============================================================================
# CONVERSIÓN DE UNIDADES
# ============================================================================

# Velocidad: m/s → km/h
v_euler_kmh = v_euler * 3.6
v_rk4_kmh = v_rk4 * 3.6
v_analitico_kmh = v_analitico * 3.6

# Aceleración: m/s² → g's
a_euler_g = a_euler / G
a_rk4_g = a_rk4 / G
a_analitico_g = a_analitico / G

# ============================================================================
# GRÁFICOS
# ============================================================================

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# ============================================================================
# GRÁFICO 1: POSICIÓN [m]
# ============================================================================

"""
ax1.plot(t_analitico, y_analitico, '-', linewidth=2.5, color='black',
         label='Analítico', alpha=0.8, zorder=1)"""
ax1.plot(t_euler, y_euler, '--', linewidth=1.5, color='blue',
         label=f'Euler (h={h_euler:.6f} s)', alpha=0.7, zorder=2)
ax1.plot(t_rk4, y_rk4, ':', linewidth=2, color='red',
         label=f'RK4 (h={h_rk4:.2f} s)', alpha=0.7, zorder=3)

ax1.axhline(y=L0, color='orange', linestyle='-.', linewidth=1.5,
            label=f'L₀ = {L0:.2f} m', alpha=0.6)

ax1.set_xlabel('Tiempo [s]', fontsize=12)
ax1.set_ylabel('Posición [m]', fontsize=12)
ax1.set_title('Posición vs Tiempo - 4 Caídas Sucesivas', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.legend(fontsize=10, loc='best')
ax1.set_xlim([0, min(t_euler[-1], t_rk4[-1])])

# ============================================================================
# GRÁFICO 2: VELOCIDAD [km/h]
# ============================================================================

"""
ax2.plot(t_analitico, v_analitico_kmh, '-', linewidth=2.5, color='black',
         label='Analítico', alpha=0.8, zorder=1)"""
ax2.plot(t_euler, v_euler_kmh, '--', linewidth=1.5, color='blue',
         label=f'Euler (h={h_euler:.6f} s)', alpha=0.7, zorder=2)
ax2.plot(t_rk4, v_rk4_kmh, ':', linewidth=2, color='red',
         label=f'RK4 (h={h_rk4:.2f} s)', alpha=0.7, zorder=3)

ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

ax2.set_xlabel('Tiempo [s]', fontsize=12)
ax2.set_ylabel('Velocidad [km/h]', fontsize=12)
ax2.set_title('Velocidad vs Tiempo - 4 Caídas Sucesivas', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.legend(fontsize=10, loc='best')
ax2.set_xlim([0, min(t_euler[-1], t_rk4[-1])])

# ============================================================================
# GRÁFICO 3: ACELERACIÓN [g's]
# ============================================================================

"""
ax3.plot(t_analitico, a_analitico_g, '-', linewidth=2.5, color='black',
         label='Analítico', alpha=0.8, zorder=1)"""
ax3.plot(t_euler, a_euler_g, '--', linewidth=1.5, color='blue',
         label=f'Euler (h={h_euler:.6f} s)', alpha=0.7, zorder=2)
ax3.plot(t_rk4, a_rk4_g, ':', linewidth=2, color='red',
         label=f'RK4 (h={h_rk4:.2f} s)', alpha=0.7, zorder=3)

ax3.axhline(y=1, color='orange', linestyle='-.', linewidth=1.5,
            label='1 g (gravedad)', alpha=0.6)
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

ax3.set_xlabel('Tiempo [s]', fontsize=12)
ax3.set_ylabel('Aceleración [g]', fontsize=12)
ax3.set_title('Aceleración vs Tiempo - 4 Caídas Sucesivas', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle=':')
ax3.legend(fontsize=10, loc='best')
ax3.set_xlim([0, min(t_euler[-1], t_rk4[-1])])

plt.tight_layout()
plt.savefig('comparacion_4_caidas.png', dpi=300, bbox_inches='tight')
plt.show()

print("Gráfico guardado: comparacion_4_caidas.png")
print()

# ============================================================================
# ANÁLISIS
# ============================================================================

print("=" * 70)
print("ANÁLISIS COMPARATIVO")
print("=" * 70)
print()

# Encontrar los primeros 4 mínimos
def encontrar_minimos(t, y, v, num=4):
    minimos_t = []
    minimos_y = []
    v_anterior = 0
    for i in range(1, len(v)):
        if v_anterior > 0 and v[i] < 0:
            minimos_t.append(t[i])
            minimos_y.append(y[i])
            if len(minimos_t) >= num:
                break
        v_anterior = v[i]
    return minimos_t, minimos_y

t_min_euler, y_min_euler = encontrar_minimos(t_euler, y_euler, v_euler)
t_min_rk4, y_min_rk4 = encontrar_minimos(t_rk4, y_rk4, v_rk4)
t_min_anal, y_min_anal = encontrar_minimos(t_analitico, y_analitico, v_analitico)

print("Puntos más bajos de las 4 primeras caídas:")
print()
print("Analítico:")
for i, (t, y) in enumerate(zip(t_min_anal, y_min_anal), 1):
    print(f"  Caída {i}: t = {t:6.2f} s, y = {y:7.2f} m")

print("\nEuler:")
for i, (t, y) in enumerate(zip(t_min_euler, y_min_euler), 1):
    if i <= len(y_min_anal):
        error = abs(y - y_min_anal[i-1]) / y_min_anal[i-1] * 100
        print(f"  Caída {i}: t = {t:6.2f} s, y = {y:7.2f} m (error = {error:.3f}%)")

print("\nRK4:")
for i, (t, y) in enumerate(zip(t_min_rk4, y_min_rk4), 1):
    if i <= len(y_min_anal):
        error = abs(y - y_min_anal[i-1]) / y_min_anal[i-1] * 100
        print(f"  Caída {i}: t = {t:6.2f} s, y = {y:7.2f} m (error = {error:.3f}%)")

print()
print("=" * 70)