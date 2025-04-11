import os

# -------------------------------------------------------------
# USO DE CHATGPT PARA LA LECTURA CORRECTA DEL BIBTEXT
# -------------------------------------------------------------

def read_bibtex(filename):
    """Leer un archivo BibTeX y convertirlo en una lista de diccionarios."""
    articles = []
    try:
        with open(filename, mode="r", encoding="utf-8") as file:
            current_article = {}
            for line in file:
                line = line.strip()
                if line.startswith("@article"):
                    current_article = {"source_file": filename}  # Guardar el archivo de origen
                elif line == "}":
                    if current_article:
                        articles.append(current_article)
                else:
                    # Parsear las líneas clave-valor
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip("{}").strip(",")
                        current_article[key] = value
    except Exception as e:
        print(f"Error al leer el archivo {filename}: {e}")
    return articles

# -----------------------------------------------------------------------------
# USO DE CHATGPT PARA INVESTIGAR EL MANEJO DE ARCHIVOS DUPLICADOS Y UNIFICAR
# -----------------------------------------------------------------------------


def unify_results_from_files(*filenames):
    """Unificar resultados a partir de varios archivos BibTeX."""
    datasets = [read_bibtex(filename) for filename in filenames]

    unique_articles = {}
    duplicates = {}

    for dataset in datasets:
        for article in dataset:
            if "title" not in article:
                print(f"Artículo sin título: {article}")
                continue

            key = article["title"].strip().lower()
            if key in unique_articles:
                if key not in duplicates:
                    duplicates[key] = {"article": article, "files": [unique_articles[key]["source_file"]]}
                duplicates[key]["files"].append(article["source_file"])
            else:
                unique_articles[key] = article

    # Crear la carpeta "Data" si no existe
    if not os.path.exists("Data"):
        os.makedirs("Data")

    # Guardar resultados unificados y duplicados
    save_bibtex("Data/unificados.bib", unique_articles.values())
    save_duplicates("Data/duplicados.bib", duplicates)

# -------------------------------------------------------------
# USO DE CHATGPT PARA LA ESTRUCTURA DE GUARDADO
# -------------------------------------------------------------

def save_bibtex(filename, articles):
    """Guardar artículos en formato BibTeX."""
    try:
        with open(filename, mode="w", encoding="utf-8") as file:
            for i, article in enumerate(articles):
                title = article.get("title", "Unknown Title")
                authors = article.get("author", "Unknown Authors")
                year = article.get("year", "Unknown Year")
                journal = article.get("journal", "Unknown Journal")
                tipo = article.get("tipo", "Unknown Type")
                publisher = article.get("publisher", "Unkown Publisher")
                abstract = article.get("abstract", "Unknown Abstract")
                url = article.get("url", "Unknown URL")

                file.write(f"@article{{ref{i},\n")
                file.write(f"  title = {{{title}}},\n")
                file.write(f"  author = {{{authors}}},\n")
                file.write(f"  year = {{{year}}},\n")
                file.write(f"  journal = {{{journal}}},\n")
                file.write(f"  tipo = {{{tipo}}},\n")
                file.write(f"  publisher = {{{publisher}}},\n")
                file.write(f"  abstract = {{{abstract}}},\n")
                file.write(f"  url = {{{url}}}\n")
                file.write("}\n\n")

        print(f"Archivo guardado correctamente: {filename}")
    except Exception as e:
        print(f"Error al guardar el archivo {filename}: {e}")

# -------------------------------------------------------------
# USO DE CHATGPT PARA LA ESTRUCTURA DE GUARDADO
# -------------------------------------------------------------

def save_duplicates(filename, duplicates):
    """Guardar duplicados en formato BibTeX con información de las páginas compartidas."""
    try:
        with open(filename, mode="w", encoding="utf-8") as file:
            for i, (key, data) in enumerate(duplicates.items()):
                article = data["article"]
                files = ", ".join(data["files"])

                title = article.get("title", "Unknown Title")
                authors = article.get("author", "Unknown Authors")
                year = article.get("year", "Unknown Year")
                journal = article.get("journal", "Unknown Journal")
                tipo = article.get("tipo", "Unknown Type")
                publisher = article.get("publisher", "Unkown Publisher")
                abstract = article.get("abstract", "Unknown Abstract")
                url = article.get("url", "Unknown URL")

                file.write(f"@article{{ref_dup{i},\n")
                file.write(f"  title = {{{title}}},\n")
                file.write(f"  author = {{{authors}}},\n")
                file.write(f"  year = {{{year}}},\n")
                file.write(f"  journal = {{{journal}}},\n")
                file.write(f"  tipo = {{{tipo}}},\n")
                file.write(f"  publisher = {{{publisher}}},\n")
                file.write(f"  abstract = {{{abstract}}},\n")
                file.write(f"  url = {{{url}}},\n")
                file.write(f"  shared_files = {{{files}}}\n")
                file.write("}\n\n")

        print(f"Archivo de duplicados guardado correctamente: {filename}")
    except Exception as e:
        print(f"Error al guardar el archivo {filename}: {e}")


# Pasamos los archivos bib con los datos para crear un solo archivo "Unificados"
unify_results_from_files("Data/resultados_ACM.bib", "Data/resultados_ieee.bib", "Data/resultados_springer_open.bib", "Data/resultados_Sage.bib")
