import time
import os
import re

# Se define la ruta del archivo BibTeX a partir del directorio actual del script
file_name = os.path.join(os.path.dirname(__file__), "unificados.bib")

# ----------------------------------------------
# Funciones de diferentes métodos de ordenamiento
# ----------------------------------------------

# CombSort: Método de ordenamiento que utiliza un "gap" decreciente para comparar y ordenar elementos.
def comb_sort(data):
    n = len(data)
    gap = n
    shrink = 1.3  # Factor de reducción del gap
    sorted = False
    while not sorted:
        gap = int(gap / shrink)  # Reducir el gap en cada iteración
        if gap <= 1:
            gap = 1
            sorted = True
        i = 0
        while i + gap < n:  # Comparar elementos separados por el gap
            if data[i] > data[i + gap]:
                # Intercambio de elementos si están fuera de orden
                data[i], data[i + gap] = data[i + gap], data[i]
                sorted = False
            i += 1

# SelectionSort: Selecciona el menor elemento y lo coloca al inicio.
def selection_sort(data):
    for i in range(len(data)):
        min_idx = i  # Índice del elemento más pequeño
        for j in range(i + 1, len(data)):
            if data[j] < data[min_idx]:
                min_idx = j  # Actualizar el índice del mínimo
        # Intercambiar el elemento más pequeño con el actual
        data[i], data[min_idx] = data[min_idx], data[i]

# TreeSort: Usa un árbol binario de búsqueda para ordenar elementos.
def tree_sort(data):
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

    # Inserta un nodo en el árbol binario de búsqueda de forma iterativa.
    def insert_iterative(root, key):
        if root is None:
            return Node(key)
        current = root
        while True:
            if key < current.key:
                if current.left is None:
                    current.left = Node(key)
                    break
                current = current.left
            else:
                if current.right is None:
                    current.right = Node(key)
                    break
                current = current.right
        return root

    # Recorrido inorden para obtener elementos ordenados.
    def inorder_traversal(root, res):
        stack = []
        current = root
        while stack or current:
            while current:
                stack.append(current)
                current = current.left
            current = stack.pop()
            res.append(current.key)
            current = current.right

    root = None
    for val in data:
        root = insert_iterative(root, val)  # Crear árbol binario
    res = []
    inorder_traversal(root, res)  # Recorrer el árbol
    return res

# BitonicSort: Método especializado que utiliza una secuencia bitónica.
def bitonic_sort(arr):
    def bitonic_merge(start, length, direction):
        if length > 1:
            mid = length // 2
            for i in range(start, start + mid):
                # Comparación y ordenamiento en función de la dirección
                if (direction and arr[i] > arr[i + mid]) or (not direction and arr[i] < arr[i + mid]):
                    arr[i], arr[i + mid] = arr[i + mid], arr[i]
            bitonic_merge(start, mid, direction)
            bitonic_merge(start + mid, mid, direction)

    def bitonic_sort_recursive(start, length, direction):
        if length > 1:
            mid = length // 2
            # Crear secuencia ascendente y descendente
            bitonic_sort_recursive(start, mid, True)
            bitonic_sort_recursive(start + mid, mid, False)
            bitonic_merge(start, length, direction)

    n = len(arr)
    # Ajustar el tamaño del arreglo a una potencia de 2
    next_power_of_two = 1 << (n - 1).bit_length()
    if n < next_power_of_two:
        arr.extend([float('inf')] * (next_power_of_two - n))  # Rellenar con infinitos
    
    # Ordenar utilizando Bitonic Sort
    bitonic_sort_recursive(0, len(arr), True)

    # Eliminar elementos extras añadidos
    while len(arr) > n:
        arr.pop()


# PigeonholeSort
def pigeonhole_sort(data):
    # Validación: verifica que los datos no sean nulos y contengan solo números
    if not data or not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("Pigeonhole Sort solo admite datos numéricos.")
    
    # Encuentra los valores mínimo y máximo en el conjunto de datos
    min_val = min(data)
    max_val = max(data)
    
    # Calcula el tamaño del rango necesario para los "agujeros"
    size = int(max_val - min_val + 1)
    holes = [0] * size  # Inicializa los agujeros como un arreglo de ceros

    # Distribuye los elementos en los "agujeros"
    for x in data:
        holes[int(x - min_val)] += 1

    # Reconstruye el arreglo ordenado desde los agujeros
    i = 0
    for count in range(size):
        while holes[count] > 0:
            data[i] = count + min_val
            i += 1
            holes[count] -= 1

# BucketSort
def bucket_sort(data):
    # Verifica si el conjunto de datos está vacío
    if len(data) == 0:
        return
    
    # Encuentra el valor máximo y calcula el tamaño del intervalo
    max_val = max(data)
    size = max_val / len(data)
    
    # Crea las cubetas como listas vacías
    buckets = [[] for _ in range(len(data))]

    # Distribuye los elementos en las cubetas correspondientes
    for x in data:
        index = int(x / size)
        if index != len(data):
            buckets[index].append(x)
        else:
            buckets[len(data) - 1].append(x)

    # Ordena cada cubeta individualmente
    for bucket in buckets:
        bucket.sort()

    # Combina los elementos ordenados de todas las cubetas
    result = []
    for bucket in buckets:
        result.extend(bucket)

    # Copia los resultados ordenados de nuevo en el arreglo original
    data[:] = result

# QuickSort
def quick_sort(data):
    # Condición base: si el tamaño es 1 o menor, ya está ordenado
    if len(data) <= 1:
        return data
    
    # Selecciona un pivote y divide el arreglo en tres partes
    pivot = data[len(data) // 2]
    left = [x for x in data if x < pivot]
    middle = [x for x in data if x == pivot]
    right = [x for x in data if x > pivot]
    
    # Aplica recursión en las partes izquierda y derecha
    return quick_sort(left) + middle + quick_sort(right)

# HeapSort
def heap_sort(data):
    import heapq  # Importa la biblioteca para manejar montículos
    # Convierte el arreglo en un montículo
    heapq.heapify(data)
    
    # Extrae los elementos en orden ascendente
    data[:] = [heapq.heappop(data) for _ in range(len(data))]

# GnomeSort
def gnome_sort(data):
    index = 0  # Inicializa el índice en 0
    while index < len(data):
        # Avanza si el elemento actual está en el orden correcto
        if index == 0 or data[index] >= data[index - 1]:
            index += 1
        else:
            # Si no, intercambia los elementos y retrocede
            data[index], data[index - 1] = data[index - 1], data[index]
            index -= 1

# BinaryInsertionSort
def binary_insertion_sort(data):
    for i in range(1, len(data)):
        key = data[i]  # Elemento a insertar
        low, high = 0, i - 1
        
        # Encuentra la posición de inserción usando búsqueda binaria
        while low <= high:
            mid = (low + high) // 2
            if data[mid] > key:
                high = mid - 1
            else:
                low = mid + 1
        
        # Inserta el elemento en la posición encontrada
        data[:] = data[:low] + [key] + data[low:i] + data[i + 1:]

# RadixSort
def radix_sort(data):
    RADIX = 10  # Base decimal
    max_digit = max(data)  # Encuentra el número más grande
    exp = 1  # Inicia con el dígito menos significativo

    # Repite el proceso mientras haya dígitos
    while max_digit // exp > 0:
        buckets = [[] for _ in range(RADIX)]  # Crea 10 cubetas para cada dígito

        # Distribuye los números en las cubetas según el dígito actual
        for i in data:
            buckets[(i // exp) % RADIX].append(i)
        
        # Reconstruye el arreglo a partir de las cubetas
        data[:] = [item for bucket in buckets for item in bucket]
        
        # Pasa al siguiente dígito más significativo
        exp *= RADIX


# ----------------------------------------------
# Leer archivo BibTeX
# ----------------------------------------------
def read_bibtex(file_name):
    data = []
    with open(file_name, "r", encoding="utf-8") as file:
        article = {}
        for line in file:
            line = line.strip()
            if line.startswith("@article"):
                article = {}  # Iniciar un nuevo artículo
            elif line.startswith("}"):
                if article:  # Guardar el artículo actual si no está vacío
                    data.append(article)
            else:
                # Separar clave y valor del campo del artículo
                key_value = line.split("=", 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip().strip("{}").strip(",")
                    article[key] = value
    return data

# ----------------------------------------------
# Calcular tiempos para diferentes métodos
# ----------------------------------------------
def analyze_sorting_total_time(articles, numeric_methods, general_methods):
    results = {}

    print("Procesando métodos para datos numéricos...")
    numeric_field_data = []
    for article in articles:
        if "year" in article:
            raw_year = str(article["year"]).strip()
            cleaned_year = re.sub(r"[^\d]", "", raw_year)  # Limpiar valores no numéricos
            if re.match(r"^\d{4}$", cleaned_year):  # Validar formato de año
                numeric_field_data.append(int(cleaned_year))

    # Procesar métodos de ordenamiento en campo "year"
    if not numeric_field_data:
        print("No se encontraron datos numéricos válidos en el campo 'year'.")
    else:
        for name, method in numeric_methods.items():
            try:
                print(f"Ejecutando {name} para el campo 'year'...")
                start_time = time.time()
                method(numeric_field_data.copy())  # Ordenar copia de los datos
                end_time = time.time()
                if name not in results:
                    results[name] = {}
                results[name]["year"] = end_time - start_time
            except Exception as e:
                print(f"Error al ordenar con '{name}': {e}")

    # Procesar métodos para todos los campos generales
    print("Procesando métodos para todos los campos...")
    for name, method in general_methods.items():
        if name not in results:
            results[name] = {}
        
        for field in ["title", "author", "year", "journal", "abstract", "url"]:
            field_data = [article[field] for article in articles if field in article and article[field]]
            
            if not field_data:
                print(f"Campo '{field}' omitido por falta de datos válidos.")
                continue
            
            try:
                # Convertir todos los valores a minúsculas para uniformidad
                if all(isinstance(val, str) for val in field_data):
                    field_data = [val.lower() for val in field_data]
                elif all(isinstance(val, (int, float)) for val in field_data):
                    field_data = list(map(float, field_data))
                else:
                    raise ValueError(f"Datos en '{field}' no son comparables.")
            except Exception as e:
                print(f"Error procesando el campo '{field}': {e}")
                continue
            
            try:
                print(f"Ejecutando {name} para el campo '{field}'...")
                start_time = time.time()
                method(field_data.copy())  # Ordenar copia de los datos
                end_time = time.time()
                results[name][field] = end_time - start_time
            except Exception as e:
                print(f"Error al ordenar con '{name}' en el campo '{field}': {e}")
                continue

    return results

# Mostrar resultados en formato tabular
def display_results(results):
    print(f"{'Método':<25} {'Campo':<15} {'Tiempo (s)':<15}")
    print("-" * 60)
    for method, fields in results.items():
        for field, time_taken in fields.items():
            print(f"{method:<25} {field:<15} {time_taken:<15.6f}")

# ----------------------------------------------
# Main: Punto de entrada del programa
# ----------------------------------------------
if __name__ == "__main__":
    articles = read_bibtex(file_name)  # Leer archivo BibTeX
    n = len(articles)

    # Métodos especializados para datos numéricos
    numeric_methods = {
        "Pigeonhole Sort": pigeonhole_sort,
        "RadixSort": radix_sort,
        "Bucket Sort": bucket_sort,
        "Bitonic Sort": bitonic_sort,
    }

    # Métodos generales de ordenamiento
    general_methods = {
        "TimSort (Python built-in)": sorted,
        "Comb Sort": comb_sort,
        "Selection Sort": selection_sort,
        "Tree Sort": lambda data: tree_sort(data),
        "QuickSort": lambda data: quick_sort(data),
        "HeapSort": heap_sort,
        "Gnome Sort": gnome_sort,
        "Binary Insertion Sort": binary_insertion_sort,
    }

    print(f"Tamaño de los datos: {n}")
    print("Resultados de ordenamiento:")
    print("-----------------------------------")
    total_times = analyze_sorting_total_time(articles, numeric_methods, general_methods)
    display_results(total_times)
