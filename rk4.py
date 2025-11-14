import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARÁMETROS DEL PROBLEMA
# ============================================================================

NP = 111687
H = 150
g = 9.81

L0 = (0.1/10000 * (NP - 100000) + 0.25) * H
m = 40/10000 * (NP - 100000) + 50
k1 = 10/10000 * (NP - 100000) + 40

print("=" * 70)
print("MÉTODO DE RUNGE-KUTTA DE ORDEN 4 (RK4)")
print("=" * 70)
print(f"\nParámetros (NP = {NP}):")
print(f"  L0 = {L0:.2f} m")
print(f"  m = {m:.2f} kg")
print(f"  k1 = {k1:.2f} N/m")
print()

# Valor analítico
mg_k1 = m * g / k1
y_max_analitico = L0 + mg_k1 + np.sqrt(mg_k1**2 + 2*m*g*L0/k1)
print(f"y_max analítico = {y_max_analitico:.2f} m\n")

# ============================================================================
# FUNCIÓN DE ACELERACIÓN
# ============================================================================

def aceleracion(y):
    if y <= L0:
        return g
    else:
        return g - (k1/m) * (y - L0)

# ============================================================================
# MÉTODO RK4
# ============================================================================

def rk4_bungee(h, t_max=20):
    t = 0.0
    y = 0.0
    v = 0.0
    
    t_list = [t]
    y_list = [y]
    v_list = [v]
    
    y_max = 0.0
    v_anterior = 0.0
    
    while t < t_max:
        # Coeficientes RK4
        k1_y = v
        k1_v = aceleracion(y)
        
        k2_y = v + 0.5 * h * k1_v
        k2_v = aceleracion(y + 0.5 * h * k1_y)
        
        k3_y = v + 0.5 * h * k2_v
        k3_v = aceleracion(y + 0.5 * h * k2_y)
        
        k4_y = v + h * k3_v
        k4_v = aceleracion(y + h * k3_y)
        
        # Actualización
        y = y + (h/6.0) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
        v = v + (h/6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        t = t + h
        
        t_list.append(t)
        y_list.append(y)
        v_list.append(v)
        
        # Detectar máximo
        if v_anterior > 0 and v < 0:
            y_max = max(y_list[-2], y)
            break
        
        if y > y_max:
            y_max = y
        
        v_anterior = v
    
    return np.array(t_list), np.array(y_list), np.array(v_list), y_max

# ============================================================================
# ANÁLISIS DEL ORDEN DE CONVERGENCIA
# ============================================================================

print("=" * 70)
print("ANÁLISIS DEL ORDEN DE CONVERGENCIA")
print("=" * 70)

# Valores de h para el análisis
h_values = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1,
                     0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02, 0.015, 0.01,
                     0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001])

print(f"\n{'h (s)':<12} {'y_max (m)':<12} {'Error (%)':<12}")
print("-" * 40)

errors = []
for h in h_values:
    _, _, _, y_max_num = rk4_bungee(h)
    error = abs(y_max_num - y_max_analitico) / y_max_analitico * 100
    errors.append(error)
    
    if h in [1.0, 0.5, 0.1, 0.05, 0.01, 0.001]:
        print(f"{h:<12.6f} {y_max_num:<12.4f} {error:<12.6f}")

errors = np.array(errors)

# Ajuste del orden (filtrar errores razonables)
mask = (errors > 1e-8) & (errors < 5.0)
h_fit = h_values[mask]
error_fit = errors[mask]

log_h = np.log10(h_fit)
log_error = np.log10(error_fit)
coef = np.polyfit(log_h, log_error, 1)
p = coef[0]

log_error_pred = np.polyval(coef, log_h)
r2 = 1 - np.sum((log_error - log_error_pred)**2) / np.sum((log_error - np.mean(log_error))**2)

print()
print(f"Orden experimental: p = {p:.2f}")
print(f"Orden teórico: 4.00")
print(f"R² = {r2:.4f}")
print()

# ============================================================================
# BÚSQUEDA DEL h ÓPTIMO (error <= 0.1%)
# ============================================================================

print("=" * 70)
print("BÚSQUEDA DEL h ÓPTIMO (error ≤ 0.1%)")
print("=" * 70)

print(f"\n{'h (s)':<12} {'y_max (m)':<12} {'Error (%)':<12} {'Estado':<15}")
print("-" * 56)

h_candidatos = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 
                         0.3, 0.4, 0.5, 0.6, 0.8, 1.0])

h_optimo = None
error_optimo = None

for h in h_candidatos:
    _, _, _, y_max_num = rk4_bungee(h)
    error = abs(y_max_num - y_max_analitico) / y_max_analitico * 100
    
    cumple = error <= 0.1
    estado = "Cumple" if cumple else "No cumple"
    
    print(f"{h:<12.6f} {y_max_num:<12.4f} {error:<12.6f} {estado}")
    
    if cumple:
        h_optimo = h
        error_optimo = error
        y_max_optimo = y_max_num

print()
if h_optimo:
    print(f"h óptimo = {h_optimo:.6f} s")
    print(f"Error = {error_optimo:.6f}%")
    print(f"y_max = {y_max_optimo:.4f} m")
else:
    print("No se encontró h óptimo")
print()

# ============================================================================
# GRÁFICO 1: ORDEN DE CONVERGENCIA
# ============================================================================

plt.figure(figsize=(10, 7))

# Error numérico (línea + puntos)
plt.loglog(h_values, errors, '-o', markersize=8, label='Error numérico',
           color='blue', alpha=0.7, linewidth=2, markeredgecolor='darkblue', 
           markeredgewidth=0.8)

# Ajuste experimental
h_line = np.logspace(np.log10(h_fit.min()), np.log10(h_fit.max()), 100)
error_line = 10**(np.polyval(coef, np.log10(h_line)))
plt.loglog(h_line, error_line, '--', linewidth=2.5, color='orange',
           label=f'Ajuste: p = {p:.2f}', alpha=0.8)

# Pendiente teórica
h_ref = np.sqrt(h_fit.min() * h_fit.max())
error_ref = 10**(np.polyval(coef, np.log10(h_ref)))
error_teorica = error_ref * (h_line / h_ref)**4
plt.loglog(h_line, error_teorica, '-.', linewidth=2.5, color='purple',
           label='Pendiente teórica = 4.0', alpha=0.7)

# Error objetivo
plt.axhline(y=0.1, color='green', linestyle='--', linewidth=2,
            label='Error objetivo = 0.1%', alpha=0.7)

# h óptimo
if h_optimo:
    plt.plot(h_optimo, error_optimo, 'o', markersize=12,
             label=f'h óptimo = {h_optimo:.4f} s', color='red', 
             zorder=5, markeredgecolor='darkred', markeredgewidth=1.5)

plt.xlabel('Paso temporal h [s]', fontsize=13)
plt.ylabel('Error relativo [%]', fontsize=13)
plt.title('Orden de Convergencia - Método RK4', fontsize=14)
plt.grid(True, which='both', alpha=0.3, linestyle='--')
plt.legend(fontsize=10, loc='upper left', framealpha=0.95)

text_str = (f'p = {p:.2f}\n'
            f'R² = {r2:.4f}')
plt.text(0.98, 0.05, text_str, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment='bottom', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9,
                  edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('rk4_orden.png', dpi=300, bbox_inches='tight')
plt.show()

print("Gráfico guardado: rk4_orden.png\n")

# ============================================================================
# GRÁFICO 2: TRAYECTORIA
# ============================================================================

if h_optimo:
    t, y, v, y_max = rk4_bungee(h_optimo, t_max=15)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Posición
    ax1.plot(t, y, linewidth=2.5, label='y(t)', color='blue')
    ax1.axhline(y=L0, color='orange', linestyle='--', linewidth=2,
                label=f'L₀ = {L0:.2f} m', alpha=0.7)
    ax1.axhline(y=y_max_analitico, color='red', linestyle='--', linewidth=2,
                label=f'y_max (analítico) = {y_max_analitico:.2f} m', alpha=0.7)
    ax1.axhline(y=y_max, color='purple', linestyle=':', linewidth=2.5,
                label=f'y_max (RK4) = {y_max:.2f} m', alpha=0.7)
    
    ax1.set_xlabel('Tiempo [s]', fontsize=12)
    ax1.set_ylabel('Posición [m]', fontsize=12)
    ax1.set_title(f'Trayectoria - Método RK4 (h = {h_optimo} s)', fontsize=14)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.legend(fontsize=9, loc='lower right')
    ax1.set_ylim([0, 140])
    
    ax1.text(0.02, 0.98, f'Error: {error_optimo:.4f}%',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.85))
    
    # Velocidad
    ax2.plot(t, v, linewidth=2.5, color='green', label='v(t)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Tiempo [s]', fontsize=12)
    ax2.set_ylabel('Velocidad [m/s]', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle=':')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('rk4_trayectoria.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráfico guardado: rk4_trayectoria.png\n")

print("=" * 70)