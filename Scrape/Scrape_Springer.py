from playwright.sync_api import sync_playwright
import os
import re

def scrape_springer_open():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            # Acceder a SpringerOpen
            page.goto("https://www.springeropen.com/")
            page.wait_for_load_state("domcontentloaded")

            # Espera y selecciona el botón de aceptar cookies
            cookies_button = page.query_selector("button[data-cc-action='accept']")
            if cookies_button:
                cookies_button.click()
                print("Cookies aceptadas.")
                
            # Hacer clic en el botón de búsqueda
            search_button_selector = 'button[data-test="header-search-button"]'
            page.wait_for_selector(search_button_selector, timeout=10000)
            page.click(search_button_selector)

            # Campo de búsqueda
            search_input_selector = 'input[data-test="search-input"]'
            page.wait_for_selector(search_input_selector, timeout=10000)
            page.fill(search_input_selector, "computational thinking")
            page.press(search_input_selector, "Enter")

            # Esperar que los resultados se carguen
            page.wait_for_selector("article.c-listing__content", timeout=20000)
            articles = page.query_selector_all("article.c-listing__content")

            if not articles:
                print("No se encontraron artículos. Revisa los selectores.")
                return

            # Guardar resultados en un archivo BibTeX
            filepath = os.path.join("Data", "resultados_springer_open.bib")
            os.makedirs("Data", exist_ok=True)

            with open(filepath, mode="w", encoding="utf-8") as file:
                for i, article in enumerate(articles):
                    try:
                        # Extraer título
                        title_element = article.query_selector("h3.c-listing__title a[data-test='title-link']")
                        title = title_element.inner_text().strip() if title_element else "Unknown"

                        # Extraer enlace
                        link = title_element.get_attribute("href") if title_element else "Unknown"
                        if link and not link.startswith("http"):
                            link = f"https://www.springeropen.com{link}"

                        # Extraer autores
                        authors_element = article.query_selector("div.c-listing__authors")
                        authors = authors_element.inner_text().strip().replace("Authors:", "").strip() if authors_element else "Unknown"

                        # Extraer año
                        year_element = article.query_selector("span[data-test='published-on']")
                        if year_element:
                          # Obtener el texto del elemento
                          year_text = year_element.inner_text()
                          # Usar una expresión regular para extraer el año (cuatro dígitos consecutivos)
                          match = re.search(r'\b\d{4}\b', year_text)
                          year = match.group(0) if match else "Unknown"
                        else:
                          year = "Unknown"

                        # Extraer revista
                        journal_element = article.query_selector("em[data-test='journal-title']")
                        journal = journal_element.inner_text().strip() if journal_element else "Unknown"

                        # Extraer abstract
                        abstract_element = article.query_selector("p")
                        abstract = abstract_element.inner_text().strip() if journal_element else "Unknown"

                        # Escribir en formato BibTeX
                        file.write(f"@article{{ref{i},\n")
                        file.write(f"  title = {{{title}}},\n")
                        file.write(f"  author = {{{authors}}},\n")
                        file.write(f"  year = {{{year}}},\n")
                        file.write(f"  journal = {{{journal}}},\n")
                        file.write(f"  abstract = {{{abstract}}},\n")
                        file.write(f"  url = {{{link}}}\n")
                        file.write("}\n\n")

                    except Exception as e:
                        print(f"Error al procesar un artículo: {e}")

                print(f"Los artículos se guardaron exitosamente en {filepath}")
        except Exception as e:
            print(f"Error general: {e}")
        finally:
            browser.close()

scrape_springer_open()
