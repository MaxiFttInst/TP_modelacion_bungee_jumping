import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# PARÁMETROS DEL PROBLEMA
# ============================================================================

NP = 111687
H = 150  # Altura del puente [m]
g = 9.81  # Gravedad [m/s^2]

# Cálculo de parámetros según las fórmulas del enunciado
L0 = (0.1/10000 * (NP - 100000) + 0.25) * H  # Longitud natural de la cuerda [m]
m = 40/10000 * (NP - 100000) + 50  # Masa [kg]
k1 = 10/10000 * (NP - 100000) + 40  # Rigidez lineal [N/m]

print(f"Parámetros calculados para NP = {NP}:")
print(f"L0 = {L0:.2f} m")
print(f"m = {m:.2f} kg")
print(f"k1 = {k1:.2f} N/m")
print(f"g = {g:.2f} m/s²")
print()

# ============================================================================
# VALOR ANALÍTICO DEL PUNTO MÁS BAJO
# ============================================================================

mg_k1 = m * g / k1
y_max_analitico = L0 + mg_k1 + np.sqrt(mg_k1**2 + 2*m*g*L0/k1)

print(f"Valor analítico del punto más bajo: y_max = {y_max_analitico:.2f} m")
print()

# ============================================================================
# FUNCIÓN DE ACELERACIÓN
# ============================================================================

def aceleracion(y, L0, k1, m, g):
    """
    Calcula la aceleración según la posición.
    """
    if y <= L0:
        return g
    else:
        return g - (k1/m) * (y - L0)

# ============================================================================
# MÉTODO DE EULER
# ============================================================================

def euler_bungee(h, L0, k1, m, g, t_max=20):
    """
    Resuelve el problema de bungee jumping usando el Método de Euler.
    """
    t = 0.0
    y = 0.0
    v = 0.0
    
    t_list = [t]
    y_list = [y]
    v_list = [v]
    
    y_max = 0.0
    v_anterior = 0.0
    
    while t < t_max:
        a = aceleracion(y, L0, k1, m, g)
        
        y_new = y + h * v
        v_new = v + h * a
        t_new = t + h
        
        t_list.append(t_new)
        y_list.append(y_new)
        v_list.append(v_new)
        
        if v_anterior > 0 and v_new < 0:
            y_max = max(y, y_new)
            break
        
        if y_new > y_max:
            y_max = y_new
        
        v_anterior = v
        y = y_new
        v = v_new
        t = t_new
    
    return np.array(t_list), np.array(y_list), np.array(v_list), y_max

# ============================================================================
# BÚSQUEDA DEL h ÓPTIMO
# ============================================================================

print("=" * 60)
print("BÚSQUEDA DEL h ÓPTIMO")
print("=" * 60)

# Objetivo: error <= 0.1%
error_objetivo = 0.1

# Fase 1: Búsqueda por órdenes de magnitud
print("\nFase 1: Exploración por órdenes de magnitud")
print(f"{'h (s)':<12} {'y_max (m)':<12} {'Error (%)':<12} {'Estado':<20}")
print("-" * 60)

h_inicial = 1.0
h_actual = h_inicial
factor = 0.1  # Reducir h por factor de 10 en cada paso

h_values = []
error_values = []

# Explorar hasta encontrar un h que cumpla el criterio
while h_actual >= 1e-6:
    _, _, _, y_max_num = euler_bungee(h_actual, L0, k1, m, g)
    error = abs(y_max_num - y_max_analitico) / y_max_analitico * 100
    
    h_values.append(h_actual)
    error_values.append(error)
    
    estado = "Cumple" if error <= error_objetivo else "No cumple"
    print(f"{h_actual:<12.6f} {y_max_num:<12.2f} {error:<12.3f} {estado}")
    
    # Si cumple el criterio, podemos detener la búsqueda gruesa
    if error <= error_objetivo:
        print(f"\nEncontrado h que cumple: h = {h_actual:.6f} s (error = {error:.3f}%)")
        h_superior = h_actual
        h_inferior = h_actual / factor
        break
    
    h_actual *= factor
else:
    print("\nNo se encontró h en el rango explorado")
    h_optimo = None
    error_optimo = None

# Fase 2: Refinamiento (búsqueda más fina entre h_inferior y h_superior)
if 'h_superior' in locals():
    print("\nFase 2: Refinamiento en el rango encontrado")
    print(f"Buscando entre h = {h_inferior:.6f} y h = {h_superior:.6f}")
    print(f"{'h (s)':<12} {'y_max (m)':<12} {'Error (%)':<12}")
    print("-" * 40)
    
    # Crear una grilla más fina en el rango prometedor
    h_refinado = np.linspace(h_inferior, h_superior, 15)
    
    h_optimo = None
    error_optimo = float('inf')
    
    for h in h_refinado:
        _, _, _, y_max_num = euler_bungee(h, L0, k1, m, g)
        error = abs(y_max_num - y_max_analitico) / y_max_analitico * 100
        
        h_values.append(h)
        error_values.append(error)
        
        print(f"{h:<12.6f} {y_max_num:<12.2f} {error:<12.3f}")
        
        # Se queda con el h más grande que cumpla con el criterio
        if error <= error_objetivo and h > (h_optimo if h_optimo else 0):
            h_optimo = h
            error_optimo = error

# Ordenar los resultados para graficar
h_values = np.array(h_values)
error_values = np.array(error_values)
sorted_indices = np.argsort(h_values)
h_values = h_values[sorted_indices]
error_values = error_values[sorted_indices]

print("\n" + "=" * 60)
if h_optimo:
    print(f"RESULTADO FINAL:")
    print(f"  h óptimo = {h_optimo:.6f} s")
    print(f"  Error relativo = {error_optimo:.3f}%")
else:
    print("No se encontró h que cumpla el criterio en el rango explorado")
print("=" * 60)
print()

# ============================================================================
# GRÁFICO 1: ERROR vs h (escala log-log) - COMPROBACIÓN DEL ORDEN
# ============================================================================

plt.figure(figsize=(10, 6))
plt.loglog(h_values, error_values, 'o-', linewidth=2, markersize=6, 
           label='Error numérico', color='blue')

# Línea de referencia con pendiente 1 (orden 1)
h_ref = np.array([h_values[0], h_values[-1]])
error_ref = error_values[0] * (h_ref / h_values[0])
plt.loglog(h_ref, error_ref, '--', linewidth=2, color='red', 
           label='Pendiente 1 (orden 1)', alpha=0.7)

# Línea horizontal en 0.1% (objetivo)
plt.axhline(y=error_objetivo, color='green', linestyle='--', linewidth=2, 
            label=f'Error objetivo ({error_objetivo}%)', alpha=0.7)

# Marcar el h óptimo si existe
if h_optimo:
    plt.plot(h_optimo, error_optimo, 'ro', markersize=12, 
             label=f'h óptimo = {h_optimo:.6f} s', zorder=5)

plt.xlabel('Paso temporal h [s]', fontsize=12)
plt.ylabel('Error relativo [%]', fontsize=12)
plt.title('Comprobación experimental del orden del Método de Euler', fontsize=14)
plt.grid(True, which='both', alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('euler_orden.png', dpi=300, bbox_inches='tight')
plt.show()

print("Gráfico de orden guardado como 'euler_orden.png'")

# ============================================================================
# GRÁFICO 2: TRAYECTORIA CON h ÓPTIMO
# ============================================================================

if h_optimo:
    print(f"\nGenerando trayectoria con h óptimo = {h_optimo:.6f} s...")
    t_opt, y_opt, v_opt, y_max_num = euler_bungee(h_optimo, L0, k1, m, g, t_max=15)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Posición vs tiempo
    ax1.plot(t_opt, y_opt, linewidth=2, label='Posición y(t)', color='blue')
    ax1.axhline(y=L0, color='orange', linestyle='--', linewidth=2, 
                label=f'L₀ = {L0:.2f} m (longitud natural)', alpha=0.7)
    ax1.axhline(y=y_max_analitico, color='red', linestyle='--', linewidth=2, 
                label=f'y_max (analítico) = {y_max_analitico:.2f} m', alpha=0.7)
    ax1.axhline(y=y_max_num, color='purple', linestyle=':', linewidth=2, 
                label=f'y_max (numérico) = {y_max_num:.2f} m', alpha=0.7)
    ax1.set_xlabel('Tiempo [s]', fontsize=12)
    ax1.set_ylabel('Posición y [m]', fontsize=12)
    ax1.set_title(f'Método de Euler con h = {h_optimo:.6f} s', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    ax1.set_ylim([0, 140])
    
    # Velocidad vs tiempo
    ax2.plot(t_opt, v_opt, linewidth=2, color='green', label='Velocidad v(t)')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Tiempo [s]', fontsize=12)
    ax2.set_ylabel('Velocidad [m/s]', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('euler_trayectoria.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Gráfico de trayectoria guardado como 'euler_trayectoria.png'")
    print()
    print(f"Resumen final:")
    print(f"  - y_max (analítico): {y_max_analitico:.2f} m")
    print(f"  - y_max (numérico):  {y_max_num:.2f} m")
    print(f"  - Error relativo:    {error_optimo:.3f}%")