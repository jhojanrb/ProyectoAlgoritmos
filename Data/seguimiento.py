import time
import os
import re

file_name = os.path.join(os.path.dirname(__file__), "unificados.bib")

# Funciones de diferentes métodos de ordenamiento

#CombSort()
def comb_sort(data):
    n = len(data)
    gap = n
    shrink = 1.3
    sorted = False
    while not sorted:
        gap = int(gap / shrink)
        if gap <= 1:
            gap = 1
            sorted = True
        i = 0
        while i + gap < n:
            if data[i] > data[i + gap]:
                data[i], data[i + gap] = data[i + gap], data[i]
                sorted = False
            i += 1

#SelectionSort()
def selection_sort(data):
    for i in range(len(data)):
        min_idx = i
        for j in range(i + 1, len(data)):
            if data[j] < data[min_idx]:
                min_idx = j
        data[i], data[min_idx] = data[min_idx], data[i]

#TreeSort()
def tree_sort(data):
    class Node:
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None

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
        root = insert_iterative(root, val)
    res = []
    inorder_traversal(root, res)
    return res

#BitonicSort()
def bitonic_sort(arr):
    def bitonic_merge(start, length, direction):
        if length > 1:
            mid = length // 2
            for i in range(start, start + mid):
                if (direction and arr[i] > arr[i + mid]) or (not direction and arr[i] < arr[i + mid]):
                    arr[i], arr[i + mid] = arr[i + mid], arr[i]
            bitonic_merge(start, mid, direction)
            bitonic_merge(start + mid, mid, direction)

    def bitonic_sort_recursive(start, length, direction):
        if length > 1:
            mid = length // 2
            bitonic_sort_recursive(start, mid, True)  # Ascendente
            bitonic_sort_recursive(start + mid, mid, False)  # Descendente
            bitonic_merge(start, length, direction)

    n = len(arr)
    # Ajustar el tamaño a una potencia de 2 si no lo es
    next_power_of_two = 1 << (n - 1).bit_length()
    if n < next_power_of_two:
        arr.extend([float('inf')] * (next_power_of_two - n))
    
    # Ordenar con Bitonic Sort
    bitonic_sort_recursive(0, len(arr), True)

    # Eliminar los elementos extra que añadimos
    while len(arr) > n:
        arr.pop()


#PigeonholeSort()
def pigeonhole_sort(data):
    if not data or not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("Pigeonhole Sort solo admite datos numéricos.")
    min_val = min(data)
    max_val = max(data)
    size = int(max_val - min_val + 1)
    holes = [0] * size

    for x in data:
        holes[int(x - min_val)] += 1

    i = 0
    for count in range(size):
        while holes[count] > 0:
            data[i] = count + min_val
            i += 1
            holes[count] -= 1

#BucketSort()
def bucket_sort(data):
    if len(data) == 0:
        return
    max_val = max(data)
    size = max_val / len(data)
    buckets = [[] for _ in range(len(data))]

    for x in data:
        index = int(x / size)
        if index != len(data):
            buckets[index].append(x)
        else:
            buckets[len(data) - 1].append(x)

    for bucket in buckets:
        bucket.sort()

    result = []
    for bucket in buckets:
        result.extend(bucket)

    data[:] = result

#QuickSort()
def quick_sort(data):
    if len(data) <= 1:
        return data
    pivot = data[len(data) // 2]
    left = [x for x in data if x < pivot]
    middle = [x for x in data if x == pivot]
    right = [x for x in data if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

#HeapSort()
def heap_sort(data):
    import heapq
    heapq.heapify(data)
    data[:] = [heapq.heappop(data) for _ in range(len(data))]

#GnomeSort()
def gnome_sort(data):
    index = 0
    while index < len(data):
        if index == 0 or data[index] >= data[index - 1]:
            index += 1
        else:
            data[index], data[index - 1] = data[index - 1], data[index]
            index -= 1

#BinaryinsertionSort()
def binary_insertion_sort(data):
    for i in range(1, len(data)):
        key = data[i]
        low, high = 0, i - 1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] > key:
                high = mid - 1
            else:
                low = mid + 1
        data[:] = data[:low] + [key] + data[low:i] + data[i + 1:]

#RadixSort()
def radix_sort(data):
    RADIX = 10
    max_digit = max(data)
    exp = 1
    while max_digit // exp > 0:
        buckets = [[] for _ in range(RADIX)]
        for i in data:
            buckets[(i // exp) % RADIX].append(i)
        data[:] = [item for bucket in buckets for item in bucket]
        exp *= RADIX

# Leer archivo BibTeX
def read_bibtex(file_name):
    data = []
    with open(file_name, "r", encoding="utf-8") as file:
        article = {}
        for line in file:
            line = line.strip()
            if line.startswith("@article"):
                article = {}
            elif line.startswith("}"):
                if article:
                    data.append(article)
            else:
                key_value = line.split("=", 1)
                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip().strip("{}").strip(",")
                    article[key] = value
    return data

# Calcular tiempos
def analyze_sorting_total_time(articles, numeric_methods, general_methods):
    results = {}

    print("Procesando métodos para datos numéricos...")
    numeric_field_data = []
    for article in articles:
        if "year" in article:
            raw_year = str(article["year"]).strip()
            cleaned_year = re.sub(r"[^\d]", "", raw_year)
            if re.match(r"^\d{4}$", cleaned_year):
                numeric_field_data.append(int(cleaned_year))

    if not numeric_field_data:
        print("No se encontraron datos numéricos válidos en el campo 'year'.")
    else:
        for name, method in numeric_methods.items():
            try:
                print(f"Ejecutando {name} para el campo 'year'...")
                start_time = time.time()
                method(numeric_field_data.copy())
                end_time = time.time()
                results[name] = end_time - start_time
            except Exception as e:
                print(f"Error al ordenar con '{name}': {e}")

    print("Procesando métodos para todos los campos...")
    for name, method in general_methods.items():
        total_time = 0
        for field in ["title", "author", "year", "journal", "abstract", "url"]:
            field_data = [article[field] for article in articles if field in article and article[field]]
            
            if not field_data:
                print(f"Campo '{field}' omitido por falta de datos válidos.")
                continue
            
            try:
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
                method(field_data.copy())
                end_time = time.time()
                total_time += end_time - start_time
            except Exception as e:
                print(f"Error al ordenar con '{name}' en el campo '{field}': {e}")
                continue
        
        results[name] = total_time

    return results


# Mostrar resultados
def display_results(results):
    print(f"{'Método':<25} {'Tiempo total (s)':<15}")
    print("-" * 45)
    for method, total_time in results.items():
        print(f"{method:<25} {total_time:<15.6f}")

# Main
if __name__ == "__main__":
    articles = read_bibtex(file_name)
    n = len(articles)

    numeric_methods = {
        "Pigeonhole Sort": pigeonhole_sort,
        "RadixSort": radix_sort,
        "Bucket Sort": bucket_sort,
        "Bitonic Sort": bitonic_sort,
    }

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
