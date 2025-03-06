from playwright.sync_api import sync_playwright
import os

def scrape_acm():
    # Crear la carpeta "Data" si no existe
    if not os.path.exists("Data"):
        os.makedirs("Data")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            # Acceder a ACM Digital Library
            page.goto("https://dl.acm.org/")
            page.wait_for_load_state("domcontentloaded")  # Esperar que el DOM esté listo

            # Buscar artículos
            search_selector = "input[name='AllField']"
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")

            # Esperar que los resultados se carguen
            page.wait_for_selector(".search__item", timeout=60000)
            results = page.query_selector_all(".search__item")

            # Guardar resultados en un archivo BibTeX
            filepath = os.path.join("Data", "resultados_ACM.bib")
            with open(filepath, mode="w", encoding="utf-8") as file:
                for i, result in enumerate(results):
                    try:
                        # Validar existencia y extraer información
                        title = result.query_selector(".hlFld-Title a").inner_text() if result.query_selector(".hlFld-Title a") else "Unknown"
                        link = result.query_selector(".hlFld-Title a").get_attribute("href") if result.query_selector(".hlFld-Title a") else "Unknown"

                        authors = result.query_selector(".rlist--inline").inner_text() if result.query_selector(".rlist--inline") else "Unknown"
                        year = result.query_selector(".bookPubDate").inner_text() if result.query_selector(".bookPubDate") else "Unknown"
                        journal = result.query_selector(".issue-item__detail").inner_text() if result.query_selector(".issue-item__detail") else "Unknown"
                        abstract = result.query_selector(".issue-item__abstract").inner_text() if result.query_selector(".issue-item__abstract") else "Unknown"

                        # Escribir en formato BibTeX
                        file.write(f"@article{{ref{i},\n")
                        file.write(f"  title = {{{title}}},\n")
                        file.write(f"  author = {{{authors}}},\n")
                        file.write(f"  year = {{{year}}},\n")
                        file.write(f"  journal = {{{journal}}},\n")
                        file.write(f"  abstract = {{{abstract}}},\n")
                        file.write(f"  url = {{{'https://dl.acm.org' + link}}}\n")
                        file.write("}\n\n")
                    except Exception as e:
                        print(f"Error al procesar un resultado: {e}")

            print(f"Los artículos se guardaron exitosamente en {filepath}")
        except Exception as e:
            print(f"Error general: {e}")
        finally:
            browser.close()

# Llamar a la función
scrape_acm()
