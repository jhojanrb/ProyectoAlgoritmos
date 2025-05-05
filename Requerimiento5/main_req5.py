import re
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import unicodedata
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import csv
from sklearn.decomposition import TruncatedSVD

nltk.download('stopwords')
nltk.data.path.append("C:/Users/jhoja/AppData/Roaming/nltk_data")
nltk.download('punkt')
nltk.download('wordnet')


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


stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    """Limpieza y preprocesamiento de texto."""
    stop_words = set(stopwords.words("english"))
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^\w\s]", "", text.lower())
    tokens = simple_preprocess(text, deacc=True)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Paso 2: TF-IDF y Doc2Vec
def tfidf_similarity(abstracts):
    """Calcular similitud TF-IDF entre abstracts."""
    print("\nPreprocesando abstracts...")
    processed_abstracts = [preprocess(ab) for ab in abstracts]
    
    # Depurar abstracts vacíos
    non_empty_abstracts = [ab for ab in processed_abstracts if ab.strip()]
    if not non_empty_abstracts:
        raise ValueError("Todos los abstracts procesados están vacíos. Revisa el preprocesamiento.")
    
    # Aviso sobre abstracts vacíos
    if len(non_empty_abstracts) < len(abstracts):
        print(f"Advertencia: {len(abstracts) - len(non_empty_abstracts)} abstracts fueron eliminados por estar vacíos.")

    # Construir matriz TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limitar el vocabulario a las 5K palabras más frecuentes
        ngram_range=(1, 3),  # Incluir unigramas, bigramas y trigramas
        max_df=0.9,         # Excluir términos extremadamente comunes
        min_df=2,           # Excluir términos poco frecuentes
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(non_empty_abstracts)
    return cosine_similarity(tfidf_matrix)

def doc2vec_similarity(abstracts):
    tagged_data = [TaggedDocument(preprocess(ab).split(), [str(i)]) for i, ab in enumerate(abstracts)]
    model = Doc2Vec(vector_size=100, window=3, min_count=2, epochs=20, dm=1, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    similarity_matrix = np.zeros((len(abstracts), len(abstracts)))
    for i in range(len(abstracts)):
        sims = model.dv.most_similar(str(i), topn=len(abstracts))
        for j, sim in sims:
            similarity_matrix[i][int(j)] = sim
    return similarity_matrix

def process_in_batches(abstracts, batch_size=500):
    for i in range(0, len(abstracts), batch_size):
        yield abstracts[i:i + batch_size]



def reduce_dimensions(similarity_matrix, n_components=50):
    svd = TruncatedSVD(n_components=n_components)
    return svd.fit_transform(similarity_matrix)



# Paso 3: Agrupamiento y Evaluación
def cluster_abstracts(similarity_matrix, method="average", n_clusters=5):
    """
    Realiza el agrupamiento jerárquico y calcula el coeficiente de Silhouette.

    Args:
        similarity_matrix (array): Matriz de similitud entre abstracts.
        method (str): Método de enlace para el agrupamiento ("average", "single", "complete").
        n_clusters (int): Número de clústeres a formar.

    Returns:
        clusters (array): Etiquetas de clústeres para cada abstract.
    """
    # Convertir similitudes a distancias
    distance_matrix = np.clip(1 - similarity_matrix, 0, None)
    np.fill_diagonal(distance_matrix, 0)  # Asegurar que la diagonal contiene ceros

    clustering = AgglomerativeClustering(
        metric="precomputed", linkage=method, n_clusters=n_clusters
    )
    clusters = clustering.fit_predict(distance_matrix)

    # Calcular el coeficiente de Silhouette
    silhouette_avg = silhouette_score(distance_matrix, clusters, metric="precomputed")
    print(f"Coeficiente de Silhouette ({method}): {silhouette_avg:.3f}")

    return clusters

# Debug adicional: Mostrar un resumen del contenido procesado
def debug_processed_abstracts(abstracts, processed_abstracts):
    for i, (original, processed) in enumerate(zip(abstracts, processed_abstracts)):
        print(f"Abstract {i} original: {original[:100]}...")
        print(f"Abstract {i} procesado: {processed}")
        print("-" * 80)

def debug_similarity(similarity_matrix, abstracts, phrase):
    indices = [i for i, abstract in enumerate(abstracts) if phrase in abstract.lower()]
    print(f"Indices que contienen '{phrase}': {indices}")
    for i in indices:
        for j in indices:
            if i != j:
                print(f"Similitud entre abstract {i} y {j}: {similarity_matrix[i, j]:.3f}")

def determine_optimal_clusters(similarity_matrix, max_clusters=10):
    """Determina el número óptimo de clusters usando el método del codo."""
    from sklearn.cluster import KMeans

    distance_matrix = np.clip(1 - similarity_matrix, 0, None)
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(distance_matrix)
        inertias.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(distance_matrix, clusters, metric="euclidean")
        silhouette_scores.append(silhouette_avg)
        print(f"Clusters: {k}, Inertia: {kmeans.inertia_:.2f}, Silhouette: {silhouette_avg:.3f}")
    
    plt.figure(figsize=(8, 6))
    plt.plot(range(2, max_clusters + 1), inertias, marker="o", label="Inertia")
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker="o", label="Silhouette")
    plt.title("Método del Codo y Silhouette")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Puntuación")
    plt.legend()
    plt.savefig("optimal_clusters.png")


# Paso 4: Visualizaciones
def plot_similarity_heatmap(similarity_matrix):
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap="YlGnBu", cbar=True, annot=True, fmt=".2f")
    plt.title("Mapa de Calor de Similitudes")
    plt.xlabel("Abstracts")
    plt.ylabel("Abstracts")
    plt.savefig("similarity_heatmap_annotated.png")
    

def plot_dendrogram(similarity_matrix, method="average"):
    """Genera un dendrograma basado en la matriz de similitud."""
    distance_matrix = np.clip(1 - similarity_matrix, 0, None)
    linkage_matrix = linkage(distance_matrix, method=method)
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, leaf_rotation=90, leaf_font_size=10)
    plt.title(f"Dendrograma de Agrupamiento ({method})")
    plt.xlabel("Abstracts")
    plt.ylabel("Distancia")
    plt.savefig(f"dendrogram_{method}.png")
    plt.show()

def plot_cluster_distribution(clusters):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=clusters)
    plt.title("Distribución de Clústeres")
    plt.xlabel("Clúster")
    plt.ylabel("Cantidad de Abstracts")
    plt.savefig("cluster_distribution.png")


def compare_models(abstracts, doc_index=0, top_k=5):
    """
    Compara los modelos TF-IDF y Doc2Vec para encontrar los abstracts más similares.
    
    Args:
        abstracts (list): Lista de abstracts.
        doc_index (int): Índice del abstract de referencia.
        top_k (int): Número de resultados más similares a mostrar.
    """
    # Calcular similitudes
    tfidf_sim = tfidf_similarity(abstracts)
    doc2vec_sim = doc2vec_similarity(abstracts)
    
    # Mostrar abstract de referencia
    print(f"\nAbstract de referencia (índice {doc_index}):")
    print(abstracts[doc_index][:200] + "...\n")
    
    # Top K según TF-IDF
    tfidf_top_indices = np.argsort(-tfidf_sim[doc_index])[1:top_k + 1]  # Excluir autosimilitud
    print("\nTop similares (TF-IDF):")
    for idx in tfidf_top_indices:
        print(f"Índice {idx}: Sim={tfidf_sim[doc_index][idx]:.3f} - {abstracts[idx][:100]}...")
    
    # Top K según Doc2Vec
    doc2vec_top_indices = sorted(doc2vec_sim[doc_index], key=lambda x: x[1], reverse=True)[:top_k]
    print("\nTop similares (Doc2Vec):")
    for idx, sim in doc2vec_top_indices:
        print(f"Índice {int(idx)}: Sim={sim:.3f} - {abstracts[int(idx)][:100]}...")

    


# Verificar agrupamientos
def inspect_clusters(clusters, abstracts):
    """Inspeccionar manualmente agrupamientos generados."""
    df = pd.DataFrame({"Abstract": abstracts, "Cluster": clusters})
    for cluster_id in sorted(df["Cluster"].unique()):
        print(f"\nCluster {cluster_id}:")
        print(df[df["Cluster"] == cluster_id]["Abstract"].head(5).to_string(index=False))

def preprocess_abstracts(abstracts):
    """Preprocesa los abstracts y elimina los vacíos."""
    processed_abstracts = [preprocess(ab) for ab in abstracts]
    non_empty_indices = [i for i, ab in enumerate(processed_abstracts) if ab.strip()]  # Índices no vacíos
    non_empty_processed = [processed_abstracts[i] for i in non_empty_indices]
    non_empty_original = [abstracts[i] for i in non_empty_indices]
    return non_empty_original, non_empty_processed

def export_comparison_results(tfidf_results, doc2vec_results, output_file="comparison_results.csv"):
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Modelo", "Índice", "Similitud", "Extracto del Abstract"])
        
        for model, results in [("TF-IDF", tfidf_results), ("Doc2Vec", doc2vec_results)]:
            for idx, sim, abstract in results:
                writer.writerow([model, idx, sim, abstract])

def plot_dendrogramFinal(similarity_matrix, method="average"):
    distance_matrix = 1 - similarity_matrix  # Convertir similitudes a distancias
    linkage_matrix = linkage(distance_matrix[np.triu_indices(len(distance_matrix), k=1)], method=method)
    
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix)
    plt.title(f"Dendrograma (Método: {method})")
    plt.xlabel("Abstracts")
    plt.ylabel("Distancia")
    plt.tight_layout()
    plt.savefig("dendrogram_final.png")
    plt.show()


# Paso 5: Guardar Resultados
def save_results(abstracts, clusters, file_path="clustering_results.csv"):
    df = pd.DataFrame({"Abstract": abstracts, "Cluster": clusters})
    df.to_csv(file_path, index=False, encoding="utf-8")
    print(f"Resultados guardados en {file_path}")

def main():
    file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/duplicados.bib"
    abstracts = load_bibtex(file_path)

    # Preprocesar y filtrar abstracts
    print("\nPreprocesando abstracts...")
    abstracts, processed_abstracts = preprocess_abstracts(abstracts)

    # Calcular similitudes
    print("\nCalculando similitudes (TF-IDF)...")
    tfidf_sim = tfidf_similarity(processed_abstracts)
    print("\nCalculando similitudes (Doc2Vec)...")
    doc2vec_sim = doc2vec_similarity(processed_abstracts)

    # Reducción de dimensiones (solo para análisis, no para clustering jerárquico)
    print("\nReducción de dimensiones para análisis...")
    tfidf_sim_reduced = reduce_dimensions(tfidf_sim, n_components=50)

    # Determinar número óptimo de clústeres (opcional)
    print("\nDeterminando número óptimo de clústeres...")
    determine_optimal_clusters(tfidf_sim)

    # Clustering jerárquico
    print("\nAgrupando abstracts (TF-IDF)...")
    tfidf_clusters = cluster_abstracts(tfidf_sim, method="average")
    print("\nAgrupando abstracts (Doc2Vec)...")
    doc2vec_clusters = cluster_abstracts(doc2vec_sim, method="average")

    # Visualizar resultados
    plot_similarity_heatmap(tfidf_sim)
    plot_dendrogram(tfidf_sim, method="average")
    plot_cluster_distribution(tfidf_clusters)
    plot_dendrogramFinal(tfidf_sim, method="average")

    # Comparar modelos
    print("\nComparando modelos...")
    doc_index = 0  # Índice del abstract de referencia
    top_k = 5      # Número de similares a mostrar
    compare_models(abstracts, doc_index=doc_index, top_k=top_k)

    # Guardar resultados
    save_results(abstracts, tfidf_clusters, file_path="tfidf_clustering_results.csv")
    save_results(abstracts, doc2vec_clusters, file_path="doc2vec_clustering_results.csv")

    print("\nProceso completado. Resultados guardados.")

if __name__ == "__main__":
    main()
