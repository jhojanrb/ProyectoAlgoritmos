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
    keyword_counts = Counter()
    for abstract in abstracts:
        abstract_lower = abstract.lower()
        for category, terms in keywords_dict.items():
            for term, synonyms in terms.items():
                for synonym in synonyms:
                    if synonym.lower() in abstract_lower:
                        keyword_counts[term] += 1
                        break  # Solo contar una vez por término principal
    return keyword_counts

# Paso 4: Generar una gráfica de barras
def plot_bar_chart(keyword_counts):

    # Ordenar los items por frecuencia y tomar los 10 primeros
    sorted_items = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=False)[:15]
    keywords, counts = zip(*sorted_items)

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

# Paso 6: Generar gráfico de co-ocurrencia
def plot_cooccurrence_network(abstracts, keywords_dict):
    G = nx.Graph()
    for abstract in abstracts:
        present_keywords = set()
        for key, synonyms in keywords_dict.items():
            if any(synonym.lower() in abstract.lower() for synonym in synonyms):
                present_keywords.add(key)
        for k1 in present_keywords:
            for k2 in present_keywords:
                if k1 != k2:
                    if G.has_edge(k1, k2):
                        G[k1][k2]["weight"] += 1
                    else:
                        G.add_edge(k1, k2, weight=1)

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black")
    plt.title("Gráfico de Co-ocurrencia de Palabras Clave")
    plt.axis("off")
    plt.savefig(os.path.join(ruta_graficos, "coocurrencia_palabras_clave.png"))
    plt.close()

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
        keyword_counts = count_keywords(abstracts, keywords)
        
        if not keyword_counts:
            print("Advertencia: No se encontraron coincidencias con las palabras clave.")
            print("Verifica que los abstracts contengan los términos buscados.")
            return
        
        # Mostrar resultados
        print("\nFrecuencias de Palabras Clave:")
        for keyword, count in sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"{keyword}: {count}")
        
        # Graficar resultados
        plot_bar_chart(keyword_counts)
        generate_wordcloud(keyword_counts)
        plot_cooccurrence_network(abstracts, keywords)
        
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    bib_file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib"
    ruta_graficos = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos Requerimiento3"
    
    # Ejecutar el flujo principal
    main(bib_file_path)
    print(f"Análisis de palabras clave completado. Gráficos guardados en {ruta_graficos}.")