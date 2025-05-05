import re
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import unicodedata

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

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

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
    top_n = 100
    similarity_matrix = []
    for i in range(len(abstracts)):
        sims = model.dv.most_similar(str(i), topn=top_n)
        similarity_matrix.append(sims)  # Guardar (índice, similitud) en lugar de matriz completa
    
    return similarity_matrix

import numpy as np


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

def main():
    file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib"
    abstracts = load_bibtex(file_path)
    
    # Opción 1: Procesamiento completo (requiere recursos)
    tfidf_sim = tfidf_similarity(abstracts)
    doc2vec_sim = doc2vec_similarity(abstracts)
    
    # Opción 2: Procesamiento por lotes (recomendado para 11K docs)
    # batch_process(abstracts)
    
    # Comparar resultados para un abstract específico
    compare_models(abstracts, doc_index=0, top_k=5)

if __name__ == "__main__":
    main()