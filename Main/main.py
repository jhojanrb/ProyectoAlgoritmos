import sys
import os
import time  # Importar módulo para medir el tiempo

# Agregar la carpeta raíz al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Scrape.Scrape_IEE import scrape_ieee
from Scrape.Scrape_Springer import scrape_springer_open
from Scrape.Scrape_ACM import scrape_acm
from Scrape.Unificar import unify_results_from_files

def save_scraped_data(filename, data):
    """Guardar datos scrapeados en formato BibTeX."""
    if not data:
        print(f"Advertencia: No se encontraron datos para guardar en {filename}.")
        return

    if not os.path.exists("Data"):
        os.makedirs("Data")

    with open(f"Data/{filename}", mode="w", encoding="utf-8") as file:
        for i, article in enumerate(data):
            title = article.get("title", "Unknown Title")
            authors = article.get("author", "Unknown Authors")
            year = article.get("year", "Unknown Year")
            journal = article.get("journal", "Unknown Journal")
            abstract = article.get("abstract", "Unknown Abstract")
            url = article.get("url", "Unknown URL")

            file.write(f"@article{{ref{i},\n")
            file.write(f"  title = {{{title}}},\n")
            file.write(f"  author = {{{authors}}},\n")
            file.write(f"  year = {{{year}}},\n")
            file.write(f"  journal = {{{journal}}},\n")
            file.write(f"  abstract = {{{abstract}}},\n")
            file.write(f"  url = {{{url}}}\n")
            file.write("}\n\n")

if __name__ == "__main__":
    try:
        
        # Unificación de resultados
        print("Unificando resultados de todos los scrapers...")
        start_time = time.time()
        unify_results_from_files("Data/resultados_ieee.bib", 
                                 "Data/resultados_springer_open.bib", 
                                 "Data/resultados_ACM.bib")
        end_time = time.time()
        print(f"Unificación finalizada en {end_time - start_time:.2f} segundos.")
        print("Datos unificados y duplicados almacenados correctamente.")

    except Exception as e:
        print(f"Se produjo un error durante la ejecución: {e}")
