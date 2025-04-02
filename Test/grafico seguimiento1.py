import matplotlib.pyplot as plt
import numpy as np

# Datos
methods = [
    "TimSort (Python built-in)", "Comb Sort", "Selection Sort", 
    "Tree Sort",  "Pigeonhole Sort", "Bucket Sort", "QuickSort", 
    "HeapSort", "Bitonic Sort", "Gnome Sort",  "Binary Insertion Sort", "RadixSort"
]

times = [
    0.02513,  0.259658, 13.479823,
    0.346567, 0.004538,  0.003236, 0.119506,  0.037158,
    0.170266, 78.753573, 6.765612, 0.009978

]

# ----------------------------------------------
# USO DE CHATGPT PARA LA REALIZACION DEL GRAFICO
# ----------------------------------------------

# Escalar los tiempos para que los menores se vean más (logaritmo base 10)
scaled_times = np.log10(np.array(times) + 1)

# Crear el gráfico
plt.figure(figsize=(12, 8))
bars = plt.barh(methods, scaled_times, color='coral', edgecolor='black')

# Añadir los valores al final de cada barra
for bar, time in zip(bars, times):
    plt.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, 
             f"{time:.6f}", va='center', fontsize=10)

# Personalizar el gráfico
plt.title("Tiempos de Ejecución General Métodos de Ordenamiento", fontsize=16)
plt.xlabel("Log10(Tiempo total (s) + 1)", fontsize=12)
plt.ylabel("Métodos de Ordenamiento", fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()

# Mostrar el gráfico
plt.show()
