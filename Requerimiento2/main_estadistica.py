import re
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import unicodedata
import sys
import os

# Agregar la carpeta raíz al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Requerimiento2.graficos import generate_and_save_charts
from Requerimiento2.limpieza_normalizacion import normalize_authors, clean_journal_name, normalize_product_type, parse_large_bib
from Requerimiento2.generar_estadisticas import generate_statistics, save_statistics

def main():
    # Procesamiento del archivo
    file_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/unificados.bib"
    print(f"Procesando archivo: {file_path}")
    
    # Paso 1: Parsear el archivo BibTeX y crear el DataFrame
    entries = parse_large_bib(file_path)
    df = pd.DataFrame(entries)  # Aquí se define el DataFrame df
    
    # Paso 2: Normalización de datos
    print("\nNormalizando datos...")
    df['author'] = df['author'].apply(normalize_authors)
    df['tipo_normalizado'] = df['tipo'].apply(normalize_product_type)  # Normalización de tipos
    
    if 'journal' in df.columns:
        df['journal'] = df['journal'].apply(clean_journal_name)
    if 'publisher' in df.columns:
        df['publisher'] = df['publisher'].apply(clean_journal_name)
    
    # Limpieza de años
    if 'year' in df.columns:
        df['year'] = df['year'].astype(str).str.extract(r'(\d{4})')[0]
        valid_years = df['year'].notna()
        print(f"- Publicaciones con año válido: {valid_years.sum()}/{len(df)}")
    
    # Paso 3: Generar estadísticas
    print("\nGenerando estadísticas...")
    stats = generate_statistics(df)
    
    # Paso 4: Mostrar resumen
    print("\nResumen de estadísticas:")
    print(f"- Total publicaciones: {len(df)}")
    print("- Distribución por tipo normalizado:")
    print(df['tipo_normalizado'].value_counts().to_string())
    
    # Paso 5: Exportar resultados
    output_stats_path = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos estadisticos/estadisticas_finales.xlsx"
    folder_graficos = "C:/2025-1/Analisis Algoritmos/Proyecto/Data/Datos estadisticos"
    # Generar y guardar gráficos como imágenes
    generate_and_save_charts(stats, folder_graficos)
    save_statistics(stats, output_stats_path)
    print(f"\nEstadísticas y graficos exportadas a: {output_stats_path}")

if __name__ == "__main__":
    main()