from playwright.sync_api import sync_playwright
import os

def scrape_ieee():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            # Acceder a IEEE Xplore
            page.goto("https://ieeexplore.ieee.org/")
            page.wait_for_load_state("domcontentloaded")  # Esperar que el DOM esté listo

            # Usar el selector correcto para el campo de búsqueda
            search_selector = 'input[type="search"]'
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")

            # Esperar que los resultados se carguen
            page.wait_for_selector(".List-results-items", timeout=60000)
            results = page.query_selector_all(".List-results-items")

            # Guardar resultados en un archivo BibTeX
            filepath = os.path.join("Data", "resultados_ieee.bib")
            with open(filepath, mode="w", encoding="utf-8") as file:
                for i, result in enumerate(results):
                    try:
                        # Validar existencia de los elementos
                        if not result:
                            continue
                        title_element = result.query_selector("a.fw-bold")
                        if not title_element:
                            continue

                        # Extraer información básica
                        title = title_element.inner_text()
                        link = title_element.get_attribute("href")
                        url = f"https://ieeexplore.ieee.org{link}"

                        # Procesar autores, año y journal
                        author_element = result.query_selector(".author text-base-md-lh")  # Ajusta si es necesario
                        authors = author_element.inner_text() if author_element else "Unknown"

                        year_element = result.query_selector(".publisher-info-container")
                        year = year_element.inner_text() if year_element else "Unknown"

                        journal_element = result.query_selector(".fw-bold")
                        journal = journal_element.inner_text() if journal_element else "Unknown"

                        abstract_element = result.query_selector(".twist-container")
                        abstact = abstract_element.inner_text() if abstract_element else "Unknown"

                        # Escribir en formato BibTeX
                        file.write(f"@article{{ref{i},\n")
                        file.write(f"  title = {{{title}}},\n")
                        file.write(f"  author = {{{authors}}},\n")
                        file.write(f"  year = {{{year}}},\n")
                        file.write(f"  journal = {{{journal}}},\n")
                        file.write(f"  abstact = {{{abstact}}},\n")
                        file.write(f"  url = {{{url}}}\n")
                        file.write("}\n\n")

                        
                    except Exception as e:
                        print(f"Error al procesar un resultado: {e}")

        except Exception as e:
            print(f"Error general: {e}")
        finally:
            print("Los articulos de la base IEEE se guardaron exitosamente")
            browser.close()

scrape_ieee()
