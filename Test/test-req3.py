import bibtexparser
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import networkx as nx
import os
import sys
import re
import unicodedata
from tqdm import tqdm
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.patches as mpatches
from itertools import combinations
import matplotlib.patheffects as path_effects


# Agregar la carpeta raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Requerimiento3.categorias import keywords

# Paso 2: Definir funciones de normalización y limpieza
def parse_large_bib(file_path):
    """Analiza archivos BibTeX grandes con manejo mejorado de campos y tipos"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entry_pattern = re.compile(
        r'@(?P<type>\w+)\s*{\s*(?P<id>[^,\s]+)\s*,\s*'
        r'(?P<fields>(?:[^@]*?)\s*(?=\s*@\w+\s*{|$))',
        re.DOTALL
    )
    
    field_pattern = re.compile(
        r'(?P<key>\w+)\s*=\s*'
        r'(?P<value>\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+\}|"[^"]*"|[^,\n]+)\s*,?\s*',
        re.DOTALL
    )
    
    entries = []
    for match in tqdm(entry_pattern.finditer(content), desc="Analizando entradas"):
        entry = {
            'ENTRYTYPE': match.group('type').lower(),
            'ID': match.group('id').strip()
        }
        
        fields_content = match.group('fields')
        for field in field_pattern.finditer(fields_content):
            key = field.group('key').lower()
            value = field.group('value').strip()
            
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('utf-8')
            entry[key] = value.strip()
        
        entries.append(entry)
    
    return entries

# Paso 3: Cargar el archivo BibTeX y extraer abstracts
def load_bibtex(file_path):
    """Carga el archivo BibTeX usando nuestro parser personalizado"""
    entries = parse_large_bib(file_path)
    
    abstracts = []
    for entry in entries:
        # Buscar abstract en diferentes variaciones de campo
        abstract = entry.get('abstract') or entry.get('abstr') or entry.get('summary')
        if abstract:
            abstracts.append(abstract)
    
    print(f"\nSe encontraron {len(abstracts)} abstracts en el archivo.")
    print(f"Total de entradas en el archivo: {len(entries)}")
    
    # Depuración: mostrar algunos abstracts si no se encuentran
    if not abstracts and len(entries) > 0:
        print("\nDepuración: Mostrando las claves de la primera entrada:")
        print(list(entries[0].keys()))
    
    return abstracts

def count_keywords(abstracts, keywords_dict):
    keyword_data = []  # Lista detallada con categoría
    keyword_counts = Counter()  # Dict para conteo rápido compatible con funciones existentes

    for abstract in abstracts:
        abstract_lower = abstract.lower()
        for category, terms in keywords_dict.items():
            for term, synonyms in terms.items():
                for synonym in synonyms:
                    if synonym.lower() in abstract_lower:
                        # Actualizar lista detallada
                        found = False
                        for entry in keyword_data:
                            if entry["Término"] == term and entry["Categoría"] == category:
                                entry["Frecuencia"] += 1
                                found = True
                                break
                        if not found:
                            keyword_data.append({
                                "Término": term,
                                "Categoría": category,
                                "Frecuencia": 1
                            })
                        # Actualizar contador rápido
                        keyword_counts[term] += 1
                        break  # Solo contar una vez por término principal
    return keyword_data, keyword_counts


# Paso 4: Generar una gráfica de barras
def plot_bar_chart(keyword_counts):

    top_10 = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:20]
    # Separar claves y valores
    keywords = [item[0] for item in top_10]
    counts = [item[1] for item in top_10]

    plt.figure(figsize=(12, 6))
    plt.barh(keywords, counts, color="skyblue")

    #Agregar etiquetas al final de la barra
    for i, count in enumerate(counts):
        plt.text(count + 0.5, i, str(count), va='center', fontsize=10)

    plt.xlabel("Frecuencia")
    plt.title("Frecuencia de Palabras Clave")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_graficos, "frecuencia_palabras_clave.png"))
    plt.close()

# Paso 5: Generar nube de palabras
def generate_wordcloud(keyword_counts):
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(keyword_counts)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(os.path.join(ruta_graficos, "nube_palabras_clave.png"))
    plt.close()

# Función para leer palabras clave y categorías desde un archivo Excel
def cargarPalabras_excel(file_path):
    df = pd.read_excel(file_path)
    keywords_by_category = df.groupby("Categoría")["Término"].apply(list).to_dict()
    return keywords_by_category

# Paso 6: Generar gráfico de co-ocurrencia
def plot_cooccurrence_network(keywords_by_category, min_cooccurrence=1):
    """Generar gráfico de red de co-ocurrencia con categorías."""

    G = nx.Graph()

    # Contar co-ocurrencias
    co_occurrence = Counter()
    for category, keywords in keywords_by_category.items():
        for kw1, kw2 in combinations(keywords, 2):
            co_occurrence[(kw1, kw2)] += 1

    # Paleta de colores más variada
    category_list = list(keywords_by_category.keys())
    color_map = {cat: plt.colormaps["Set3"].resampled(len(category_list))(i) for i, cat in enumerate(category_list)}

    # Agregar nodos
    for category, keywords in keywords_by_category.items():
        for keyword in keywords:
            if not G.has_node(keyword):
                G.add_node(keyword, category=category)

    # Agregar aristas
    for (kw1, kw2), weight in co_occurrence.items():
        if weight >= min_cooccurrence:
            G.add_edge(kw1, kw2, weight=weight)

    # Calcular tamaños por grado ponderado
    degrees = dict(G.degree(weight="weight"))
    max_degree = max(degrees.values()) if degrees else 1
    sizes = {node: 100 + 1000 * (deg / max_degree) for node, deg in degrees.items()}

    # Layout
    pos = nx.spring_layout(G, k=2.2, iterations=100, seed=42)

    plt.figure(figsize=(24, 20))

    # Dibujar nodos
    for category, color in color_map.items():
        nodes = [n for n in G.nodes if G.nodes[n]["category"] == category]
        node_sizes = [sizes[n] for n in nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=node_sizes, node_color=[color]*len(nodes), alpha=0.9)

    # Dibujar aristas
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    weights_scaled = [0.5 + w*0.3 for w in weights]
    nx.draw_networkx_edges(G, pos, width=weights_scaled, edge_color="lightgray", alpha=0.6)

    # Dibujar etiquetas
    labels = {n: n for n in G.nodes()}
    text = nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    for t in text.values():
        t.set_path_effects([path_effects.withStroke(linewidth=2.5, foreground="white")])

    # Crear parches de leyenda
    legend_patches = [mpatches.Patch(color=color, label=category) for category, color in color_map.items()]
    plt.legend(handles=legend_patches, title="Categorías", loc="lower left", fontsize=10, title_fontsize=11, frameon=True)

    plt.title("Red de Co-Ocurrencia de Palabras Clave por Categorías", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(ruta_graficos, "keyword network.png"))
    plt.close()

# Paso 6: Guardar resultados en un archivo Excel

def guardar_keywords_en_excel(keyword_data, output_path):
    df = pd.DataFrame(keyword_data)
    df = df.sort_values(by=["Categoría", "Frecuencia"], ascending=[True, False])
    df.to_excel(output_path, index=False)

# Paso 7: Integrar todo el flujo
def main(bib_file_path):
    try:
        # Cargar abstracts
        abstracts = load_bibtex(bib_file_path)
        
        if not abstracts:
            print("Advertencia: No se encontraron abstracts en el archivo.")
            print("Verifica que los campos 'abstract' existan en las entradas .bib")
            return
        
        # Contar palabras clave
        keyword_data, keyword_counts = count_keywords(abstracts, keywords)
        
        if not keyword_counts:
            print("Advertencia: No se encontraron coincidencias con las palabras clave.")
            print("Verifica que los abstracts contengan los términos buscados.")
            return
        
        # Mostrar resultados
        print("\nFrecuencias de Palabras Clave:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{keyword}: {count}")

        # Guardar resultados en Excel
        output_excel = os.path.join(ruta_graficos, "frecuencia_keywords_categorizadas.xlsx")
        guardar_keywords_en_excel(keyword_data, output_excel)
        print(f"Archivo Excel guardado en: {output_excel}")

        excel_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento3/frecuencia_keywords_categorizadas.xlsx"

        # Cargar palabras clave desde el archivo Excel
        keywords_by_category = cargarPalabras_excel(excel_path)
        
        # Graficar resultados
        plot_bar_chart(keyword_counts)
        generate_wordcloud(keyword_counts)
        plot_cooccurrence_network(keywords_by_category)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    bib_file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib"
    ruta_graficos = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento3"
    
    
    # Ejecutar el flujo principal
    main(bib_file_path)
    print(f"Análisis de palabras clave completado. Gráficos guardados en {ruta_graficos}.")