import time
import os
import re

# Se define la ruta del archivo BibTeX a partir del directorio actual del script
file_name = os.path.join(os.path.dirname(__file__), "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib")

# ----------------------------------------------
# Funciones de diferentes métodos de ordenamiento
# ----------------------------------------------

# ----------------------------------------------
# USO DE CHATGPT PARA LA IMPLEMENTACION DE LOS ALGORITMOS
# ----------------------------------------------

# CombSort: Método de ordenamiento que utiliza un "gap" decreciente para comparar y ordenar elementos.
def comb_sort(data):
    n = len(data) #tamaño del arreglo
    gap = n # Inicializa el valor de gap (separación) al tamaño del arreglo.
    shrink = 1.3  # Factor de reducción del gap
    sorted = False # Inicializa el estado de "ordenado" como falso.
    while not sorted:
        gap = int(gap / shrink)  # Reducir el gap en cada iteración
        if gap <= 1:
            gap = 1 # Establece el gap a 1 (mínimo posible).
            sorted = True
        i = 0 # Inicializa el índice para iterar sobre el arreglo.
        while i + gap < n:  # Comparar elementos separados por el gap
            if data[i] > data[i + gap]: # Si el elemento actual es mayor que el elemento separado por el gap:
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
    # Define una clase interna para los nodos del árbol binario de búsqueda.
    class Node:
        def __init__(self, key):
            # Inicializa un nodo con una clave, un hijo izquierdo y un hijo derecho.
            self.key = key
            self.left = None
            self.right = None

    # Función para insertar un valor en el árbol binario de búsqueda de manera iterativa.
    def insert_iterative(root, key):
        if root is None:
            # Si el árbol está vacío, crea un nuevo nodo como raíz.
            return Node(key)
        current = root
        # Itera para encontrar la posición adecuada para el nuevo nodo.
        while True:
            if key < current.key:
                # Si la clave es menor que la clave actual, ir al subárbol izquierdo.
                if current.left is None:
                    # Si no hay nodo a la izquierda, insertar aquí.
                    current.left = Node(key)
                    break
                current = current.left
            else:
                # Si la clave es mayor o igual, ir al subárbol derecho.
                if current.right is None:
                    # Si no hay nodo a la derecha, insertar aquí.
                    current.right = Node(key)
                    break
                current = current.right
        return root  # Devuelve la raíz del árbol (puede haber cambiado).

    # Función para realizar un recorrido inorden del árbol y obtener elementos ordenados.
    def inorder_traversal(root, res):
        stack = []  # Pila para almacenar nodos mientras se navega por el árbol.
        current = root
        while stack or current:
            # Mientras haya nodos en la pila o un nodo actual:
            while current:
                # Descender al nodo más a la izquierda.
                stack.append(current)
                current = current.left
            # Procesar el nodo más a la izquierda.
            current = stack.pop()
            res.append(current.key)  # Agregar la clave del nodo a la lista de resultados.
            current = current.right  # Continuar con el subárbol derecho.

    root = None  # Inicializa el árbol como vacío.
    for val in data:
        # Inserta cada elemento de la lista en el árbol binario de búsqueda.
        root = insert_iterative(root, val)
    res = []  # Lista para almacenar los resultados ordenados.
    inorder_traversal(root, res)  # Realiza el recorrido inorden del árbol.
    return res  # Devuelve la lista de elementos ordenados.


# BitonicSort: Método especializado que utiliza una secuencia bitónica.
def bitonic_sort(arr):
    # Función para realizar la fusión bitónica en una secuencia.
    def bitonic_merge(start, length, direction):
        if length > 1:
            mid = length // 2
            for i in range(start, start + mid):
                # Comparación y ordenamiento en función de la dirección:
                # Si la dirección es ascendente (True) y el elemento actual es mayor que su par, se intercambian.
                # Si la dirección es descendente (False) y el elemento actual es menor que su par, se intercambian.
                if (direction and arr[i] > arr[i + mid]) or (not direction and arr[i] < arr[i + mid]):
                    arr[i], arr[i + mid] = arr[i + mid], arr[i]
            # Llamada recursiva para fusionar las mitades de forma bitónica.
            bitonic_merge(start, mid, direction)
            bitonic_merge(start + mid, mid, direction)

    # Función recursiva para dividir y ordenar las subsecuencias en forma bitónica.
    def bitonic_sort_recursive(start, length, direction):
        if length > 1:
            mid = length // 2
            # Crear una subsecuencia ascendente.
            bitonic_sort_recursive(start, mid, True)
            # Crear una subsecuencia descendente.
            bitonic_sort_recursive(start + mid, mid, False)
            # Fusionar las subsecuencias en una secuencia bitónica.
            bitonic_merge(start, length, direction)

    n = len(arr)  # Longitud del arreglo original.
    # Ajustar el tamaño del arreglo a la potencia de 2 más cercana superior.
    next_power_of_two = 1 << (n - 1).bit_length()
    if n < next_power_of_two:
        # Rellenar con infinitos para igualar la longitud a una potencia de 2.
        arr.extend([float('inf')] * (next_power_of_two - n))
    
    # Ordenar el arreglo utilizando el algoritmo de Bitonic Sort.
    bitonic_sort_recursive(0, len(arr), True)

    # Eliminar los elementos extras añadidos previamente (valores infinitos).
    while len(arr) > n:
        arr.pop()



# Pigeonhole Sort: Ordena elementos utilizando una estrategia de "agujeros" basados en el rango de valores.
def pigeonhole_sort(data):
    # Validación: verifica que los datos no sean nulos y que todos los elementos sean números (int o float).
    if not data or not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("Pigeonhole Sort solo admite datos numéricos.")
    
    # Encuentra el valor mínimo en el conjunto de datos.
    min_val = min(data)
    # Encuentra el valor máximo en el conjunto de datos.
    max_val = max(data)
    
    # Calcula el tamaño del rango de valores necesarios para los "agujeros".
    size = int(max_val - min_val + 1)
    # Inicializa los "agujeros" como un arreglo de ceros. Cada índice representará un valor en el rango.
    holes = [0] * size

    # Distribuye cada elemento del conjunto de datos en el agujero correspondiente.
    for x in data:
        # Calcula el índice del agujero correspondiente restando el valor mínimo.
        holes[int(x - min_val)] += 1

    # Reconstruye el arreglo ordenado desde los agujeros.
    i = 0  # Índice para sobrescribir el arreglo original.
    for count in range(size):
        # Mientras haya elementos en el agujero actual, se añaden de vuelta al arreglo.
        while holes[count] > 0:
            # Asigna el valor correspondiente al índice actual.
            data[i] = count + min_val
            i += 1  # Avanza al siguiente índice en el arreglo.
            holes[count] -= 1  # Disminuye el contador del agujero actual.


# Bucket Sort: Ordena los elementos distribuyéndolos en cubetas según su valor.
def bucket_sort(data):
    # Verifica si el conjunto de datos está vacío
    if len(data) == 0:
        return

    # Encuentra el valor máximo y calcula el tamaño de los intervalos
    max_val = max(data)
    size = max_val / len(data)

    # Crea una lista de cubetas vacías, una para cada elemento del conjunto de datos
    buckets = [[] for _ in range(len(data))]

    # Distribuye los elementos en las cubetas correspondientes
    for x in data:
        index = int(x / size)  # Determina la cubeta en la que debe ir el elemento
        if index != len(data):  # Si el índice está dentro del rango
            buckets[index].append(x)
        else:  # Si no, agrega el elemento a la última cubeta
            buckets[len(data) - 1].append(x)

    # Ordena cada cubeta individualmente
    for bucket in buckets:
        bucket.sort()

    # Combina los elementos ordenados de todas las cubetas en un solo arreglo
    result = []
    for bucket in buckets:
        result.extend(bucket)

    # Copia los resultados ordenados de nuevo en el arreglo original
    data[:] = result


# Quick Sort: Divide y conquista utilizando un pivote para ordenar.
def quick_sort(data):
    # Condición base: si el tamaño es 1 o menor, el arreglo ya está ordenado
    if len(data) <= 1:
        return data

    # Selecciona un pivote (el elemento central) y divide los datos en tres partes
    pivot = data[len(data) // 2]
    left = [x for x in data if x < pivot]  # Elementos menores que el pivote
    middle = [x for x in data if x == pivot]  # Elementos iguales al pivote
    right = [x for x in data if x > pivot]  # Elementos mayores que el pivote

    # Llama recursivamente a Quick Sort en las partes izquierda y derecha
    return quick_sort(left) + middle + quick_sort(right)


# Heap Sort: Ordena utilizando un montículo (heap).
def heap_sort(data):
    import heapq  # Biblioteca para manejar montículos (heaps)

    # Convierte el arreglo en un montículo (estructura de datos que permite extraer el mínimo rápidamente)
    heapq.heapify(data)

    # Extrae los elementos del montículo en orden ascendente
    data[:] = [heapq.heappop(data) for _ in range(len(data))]


# Gnome Sort: Ordena verificando y corrigiendo el orden de los elementos de manera iterativa.
def gnome_sort(data):
    index = 0  # Inicializa el índice en 0
    while index < len(data):
        # Si el índice es 0 o los elementos están en el orden correcto, avanza al siguiente índice
        if index == 0 or data[index] >= data[index - 1]:
            index += 1
        else:
            # Si no, intercambia los elementos actuales y retrocede un índice
            data[index], data[index - 1] = data[index - 1], data[index]
            index -= 1


# Binary Insertion Sort: Inserta elementos en su posición correcta usando búsqueda binaria.
def binary_insertion_sort(data):
    for i in range(1, len(data)):
        key = data[i]  # Elemento a insertar
        low, high = 0, i - 1  # Define el rango para la búsqueda binaria

        # Encuentra la posición de inserción usando búsqueda binaria
        while low <= high:
            mid = (low + high) // 2  # Encuentra el índice medio
            if data[mid] > key:  # Si el valor medio es mayor que el elemento clave
                high = mid - 1  # Ajusta el límite superior
            else:  # Si no, ajusta el límite inferior
                low = mid + 1

        # Inserta el elemento en la posición encontrada
        data[:] = data[:low] + [key] + data[low:i] + data[i + 1:]


# Radix Sort: Ordena los números procesando dígitos de menor a mayor significancia.
def radix_sort(data):
    RADIX = 10  # Base decimal (10 dígitos posibles: 0-9)
    max_digit = max(data)  # Encuentra el número más grande para determinar el número de dígitos
    exp = 1  # Inicializa el exponente como 1 (dígito menos significativo)

    # Repite el proceso mientras haya dígitos en el número más grande
    while max_digit // exp > 0:
        # Crea 10 cubetas (listas vacías) para agrupar números según el dígito actual
        buckets = [[] for _ in range(RADIX)]

        # Distribuye los números en las cubetas según el dígito actual
        for i in data:
            buckets[(i // exp) % RADIX].append(i)

        # Reconstruye el arreglo a partir de las cubetas
        data[:] = [item for bucket in buckets for item in bucket]

        # Incrementa el exponente para procesar el siguiente dígito más significativo
        exp *= RADIX
    

# ----------------------------------------------
# Leer archivo BibTeX CHATGPT
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
        
        for field in ["title", "author", "year", "journal"]:
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


# ----------------------------------------------CHATGPT------------------------------------------#

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
