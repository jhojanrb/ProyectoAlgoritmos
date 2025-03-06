def unify_results(*datasets):
    unique_articles = {}
    duplicates = []

    for dataset in datasets:
        for article in dataset:
            key = article["title"].lower()  # Clave para identificar duplicados
            if key in unique_articles:
                duplicates.append(article)
            else:
                unique_articles[key] = article

    # Guardar resultados unificados y duplicados
    save_bibtex("data/unificados.bib", unique_articles.values())
    save_bibtex("data/duplicados.bib", duplicates)

def save_bibtex(filename, articles):
    with open(filename, mode="w", encoding="utf-8") as file:
        for i, article in enumerate(articles):
            file.write(f"@article{{ref{i},\n")
            file.write(f"  title = {{{article['title']}}},\n")
            file.write(f"  author = {{{article['authors']}}},\n")
            file.write(f"  year = {{{article['year']}}},\n")
            file.write(f"  journal = {{{article['journal']}}},\n")
            file.write(f"  url = {{{article['url']}}}\n")
            file.write("}\n\n")
