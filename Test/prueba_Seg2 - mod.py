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
from nltk.stem import WordNetLemmatizer
import time
import os
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from scipy.sparse import vstack


nltk.download('stopwords')
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
lemmatizer = WordNetLemmatizer()

# Paso 4: Preprocesar el texto
# Mantener palabras con números (ej: "VR2") pero eliminar caracteres especiales
# y lematizar las palabras, eliminar stopwords
def preprocess(text):
    # Mantener palabras con números (ej: "VR2") pero eliminar caracteres especiales
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = simple_preprocess(text, deacc=True, min_len=3)
    # Lematización y filtrado de stopwords
    return [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]


def tfidf_similarity(abstracts):
    processed_abstracts = [' '.join(preprocess(ab)) for ab in abstracts]
    vectorizer = TfidfVectorizer(
        max_features=5000,       # Limitar el vocabulario a las 5K palabras más frecuentes
        ngram_range=(1, 3),      # Incluir unigramas, bigramas y trigramas
        stop_words='english'     # Eliminar stopwords (redundante con preprocess, pero útil)
    )
    tfidf_matrix = vectorizer.fit_transform(processed_abstracts)
    return cosine_similarity(tfidf_matrix)



def doc2vec_similarity(abstracts, save_model=True):
    tagged_data = [TaggedDocument(preprocess(ab), [str(i)]) for i, ab in enumerate(abstracts)]
    
    model = Doc2Vec(
        vector_size=100,
        min_count=2,
        epochs=20,               # Reducir epochs para velocidad (20 suele ser suficiente)
        dm=1,
        workers=4                # Paralelizar en 4 núcleos
    )
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
    
    if save_model:
        model.save("doc2vec_model.model")  # Guardar modelo para reutilizar
    
    # Calcular similitud solo para los top N más similares (ej: top 100)
    top_n = 1000
    similarity_matrix = []
    for i in range(len(abstracts)):
        sims = model.dv.most_similar(str(i), topn=top_n)
        similarity_matrix.append(sims)  # Guardar (índice, similitud) en lugar de matriz completa
    
    return similarity_matrix



# Procesar en lotes para evitar problemas de memoria
# y mejorar la velocidad de cálculo
def batch_process(abstracts, batch_size=1000): 
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        tfidf_sim_batch = tfidf_similarity(batch)
        np.save(f"tfidf_sim_batch_{i}.npy", tfidf_sim_batch)

def compare_models(abstracts, doc_index=0, top_k=5):
    # Ejemplo: Comparar los top K abstracts más similares según ambos modelos
    tfidf_sim = tfidf_similarity(abstracts)
    doc2vec_sim = doc2vec_similarity(abstracts)
    
    print(f"\nAbstract de referencia (índice {doc_index}):")
    print(abstracts[doc_index][:200] + "...")
    
    # Top K según TF-IDF
    tfidf_top = np.argsort(-tfidf_sim[doc_index])[1:top_k + 1]  # Excluir autosimilitud
    print("\nTop similares (TF-IDF):")
    for idx in tfidf_top:
        print(f"Índice {idx}: Sim={tfidf_sim[doc_index][idx]:.3f} - {abstracts[idx][:100]}...")
    
    # Top K según Doc2Vec
    print("\nTop similares (Doc2Vec):")
    for idx, sim in doc2vec_sim[doc_index][:top_k]:
        print(f"Índice {int(idx)}: Sim={sim:.3f} - {abstracts[int(idx)][:100]}...")

    #--------------------------------------------------------------------#

    # Asumiendo que ya tienes la función preprocess definida previamente
def batch_tfidf_similarity(abstracts, batch_size=500):
    """Calcula similitudes TF-IDF por lotes para ahorrar memoria."""
    tfidf_matrices = []
    for i in range(0, len(abstracts), batch_size):
        batch = abstracts[i:i + batch_size]
        processed_batch = [' '.join(preprocess(ab)) for ab in batch]
        tfidf_matrix = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english').fit_transform(processed_batch)
        tfidf_matrices.append(tfidf_matrix)

    # Combina los lotes en una matriz grande
    tfidf_combined = vstack(tfidf_matrices)
    return cosine_similarity(tfidf_combined)


# Paso 4: Crear dendrograma
# dendograma completo con todos los documentos
def create_dendrogram(similarity_matrix, labels=None):
    """Genera un dendrograma basado en una matriz de similitud."""
    # Convertir similitud a distancia
    distance_matrix = 1 - similarity_matrix

    # Forzar simetría
    distance_matrix = (distance_matrix + distance_matrix.T) / 2

    # Asegurarse de que no haya valores negativos
    distance_matrix[distance_matrix < 0] = 0

    # Corregir la diagonal
    np.fill_diagonal(distance_matrix, 0)

    # Convertir a formato adecuado para linkage
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    # Crear el dendrograma
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=None, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrograma de clustering jerárquico")
    plt.xlabel("Documentos")
    plt.ylabel("Distancia")
    plt.savefig("dendrogram.png", dpi=300, bbox_inches='tight')

# dendograma con muestra de documentos
def create_sampled_dendrogram(similarity_matrix, labels, sample_size=100):
    """
    Crea un dendrograma usando una muestra de documentos.
    :param similarity_matrix: Matriz de similitud original.
    :param labels: Etiquetas de los documentos.
    :param sample_size: Tamaño de la muestra (número de documentos a considerar).
    """
    # Validar que el tamaño de la muestra no exceda la cantidad total de documentos
    total_documents = len(labels)
    if sample_size > total_documents:
        sample_size = total_documents

    # Seleccionar una muestra aleatoria
    sampled_indices = np.random.choice(total_documents, size=sample_size, replace=False)
    sampled_similarity_matrix = similarity_matrix[np.ix_(sampled_indices, sampled_indices)]
    sampled_labels = [labels[i] for i in sampled_indices]

    # Convertir similitud a distancia
    distance_matrix = 1 - sampled_similarity_matrix
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    distance_matrix[distance_matrix < 0] = 0
    np.fill_diagonal(distance_matrix, 0)

    # Generar linkage
    linkage_matrix = linkage(squareform(distance_matrix), method='ward')

    # Crear gráfico
    plt.figure(figsize=(20, 10))
    dendrogram(linkage_matrix, labels=sampled_labels, leaf_rotation=90, leaf_font_size=10)
    plt.title("Dendrograma de clustering jerárquico (muestra)")
    plt.xlabel("Documentos")
    plt.ylabel("Distancia")
    plt.savefig("sampled_dendrogram100.png", dpi=300, bbox_inches='tight')

def save_batch_results(matrix, batch_index, output_dir):
    """Guarda resultados por lotes como archivos .npy."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"batch_{batch_index}.npy")
    np.save(file_path, matrix)

def load_batch_results(output_dir):
    """Carga y combina resultados de similitud por lotes."""
    files = sorted([f for f in os.listdir(output_dir) if f.endswith('.npy')])
    matrices = [np.load(os.path.join(output_dir, f)) for f in files]
    return np.vstack(matrices)


def main():
    file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib"
    abstracts = load_bibtex(file_path)
    print(f"Cargando y procesando {len(abstracts)} abstracts...")
    output_dir = "similarity_batches"
    
    # Opción 1: Procesamiento completo (requiere recursos)
    """""
    start_time = time.time()
    tfidf_sim = tfidf_similarity(abstracts)
    print("\nSimilitud TF-IDF calculada.")
    end_time = time.time()
    print(f"Tiempo de cálculo TF-IDF: {end_time - start_time:.2f} segundos")
    start_time = time.time()
    doc2vec_sim = doc2vec_similarity(abstracts)
    print("\nSimilitud Doc2Vec calculada.")
    end_time = time.time()
    print(f"Tiempo de cálculo Doc2Vec: {end_time - start_time:.2f} segundos")"""

    # Opción 2: Procesamiento por lotes (recomendado para 11K docs)
    # Procesar similitud por lotes y guardar resultados
    print("Calculando similitud TF-IDF por lotes...")
    start_time = time.time()
    similarity_matrix = batch_tfidf_similarity(abstracts, batch_size=500)
    save_batch_results(similarity_matrix, 0, output_dir)
    end_time = time.time()
    print(f"Tiempo de cálculo por lotes: {end_time - start_time:.2f} segundos")

    # Generar dendrograma
    print("Generando dendrograma...")
    #dendograma completo
    #create_dendrogram(similarity_matrix, labels=[f"Doc {i}" for i in range(len(abstracts))])
    #dendograma con muestra
    start_time = time.time()
    create_sampled_dendrogram(similarity_matrix, labels=[f"Doc {i}" for i in range(len(similarity_matrix))], sample_size=100)
    end_time = time.time()
    print(f"Tiempo de generación de dendrograma: {end_time - start_time:.2f} segundos")
    # batch_process(abstracts)
    
    # Comparar resultados para un abstract específico
    print("\nComparando modelos...")
    #tiempo de comparacion
    start_time = time.time()
    compare_models(abstracts, doc_index=0, top_k=10)
    end_time = time.time()
    print(f"\nTiempo de comparación: {end_time - start_time:.2f} segundos")
    

if __name__ == "__main__":
    main()