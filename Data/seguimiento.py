import time
import os

file_name = os.path.join(os.path.dirname(__file__), "unificados.bib")

# Funciones de diferentes métodos de ordenamiento
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

def selection_sort(data):
    for i in range(len(data)):
        min_idx = i
        for j in range(i + 1, len(data)):
            if data[j] < data[min_idx]:
                min_idx = j
        data[i], data[min_idx] = data[min_idx], data[i]

def tree_sort(data):
    class Node:
        def __init__(self, key):
            self.left = None
            self.right = None
            self.val = key

    def insert(root, key):
        if root is None:
            return Node(key)
        if key < root.val:
            root.left = insert(root.left, key)
        else:
            root.right = insert(root.right, key)
        return root

    def inorder_traversal(root, res):
        if root:
            inorder_traversal(root.left, res)
            res.append(root.val)
            inorder_traversal(root.right, res)

    root = None
    for val in data:
        root = insert(root, val)
    res = []
    inorder_traversal(root, res)
    return res

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

# Calcular tiempo total de cada método
def analyze_sorting_total_time(articles, methods):
    results = {}
    for name, method in methods.items():
        total_time = 0
        for field in ["title", "author", "year", "journal", "abstract", "url"]:
            field_data = [article[field] for article in articles if field in article]
            start_time = time.time()
            method(field_data.copy())  # Ordena una copia de los datos
            end_time = time.time()
            total_time += end_time - start_time
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
    n = len(articles)  # Tamaño de los datos

    sorting_methods = {
        "TimSort (Python built-in)": sorted,
        "Comb Sort": comb_sort,
        "Selection Sort": selection_sort,
        "Tree Sort": lambda data: tree_sort(data),
        # Agregar más métodos de ordenamiento aquí
    }

    print(f"Tamaño de los datos: {n}")
    print("Resultados de ordenamiento:")
    print("-----------------------------------")
    total_times = analyze_sorting_total_time(articles, sorting_methods)
    display_results(total_times)
