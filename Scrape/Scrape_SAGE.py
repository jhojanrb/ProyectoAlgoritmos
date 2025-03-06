from playwright.sync_api import sync_playwright
import os

def scrape_sage():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            # Acceder a SAGE Journals
            page.goto("https://journals.sagepub.com/")
            page.wait_for_load_state("domcontentloaded")  # Esperar que el DOM esté listo

            # Buscar artículos
            search_selector = "input#SearchTerm"
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")

            # Esperar que los resultados se carguen
            page.wait_for_selector(".search-results__item", timeout=60000)
            results = page.query_selector_all(".search-results__item")

            # Extraer datos
            data = []
            for result in results:
                try:
                    title_element = result.query_selector(".hlFld-Title a")
                    title = title_element.inner_text() if title_element else "Unknown"
                    link = title_element.get_attribute("href") if title_element else "Unknown"

                    author_element = result.query_selector(".authors")
                    authors = author_element.inner_text() if author_element else "Unknown"

                    year_element = result.query_selector(".year")
                    year = year_element.inner_text() if year_element else "Unknown"

                    journal_element = result.query_selector(".publication")
                    journal = journal_element.inner_text() if journal_element else "Unknown"

                    data.append({
                        "title": title,
                        "authors": authors,
                        "year": year,
                        "journal": journal,
                        "url": link,
                        "source": "SAGE"
                    })
                except Exception as e:
                    print(f"Error al procesar un resultado: {e}")

            return data
        finally:
            browser.close()
