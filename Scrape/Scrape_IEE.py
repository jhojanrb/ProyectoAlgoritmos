from playwright.sync_api import sync_playwright
import os
import re
import time

def scrape_ieee():
    with sync_playwright() as p:
        start_time = time.time()
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:
            # Acceder a IEEE Xplore
            page.goto("https://ieeexplore.ieee.org/")
            page.wait_for_load_state("domcontentloaded")  # Esperar que el DOM esté listo

            # Buscar el término deseado
            search_selector = 'input[type="search"]'
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")

            # Esperar que los resultados se carguen
            page.wait_for_selector(".List-results-items", timeout=60000)

            # Cambiar a mostrar 100 resultados por página
            try:
                items_per_page_button = page.locator('button:has-text("Items Per Page")')
                items_per_page_button.click()
                option_100 = page.locator('button:has-text("100")')
                option_100.click()
                page.wait_for_timeout(5000)
            except Exception as e:
                print(f"No se pudo cambiar a 100 resultados por página: {e}")

            # Crear el archivo para guardar los resultados
            os.makedirs("Data", exist_ok=True)
            filepath = os.path.join("Data", "resultados_ieee.bib")
            with open(filepath, mode="w", encoding="utf-8") as file:
                current_page = 1
                while current_page <= 20:  # Iterar hasta la página 20
                    print(f"Procesando página {current_page}...")

                    # Procesar los resultados actuales
                    results = page.query_selector_all(".List-results-items")
                    for i, result in enumerate(results):
                        try:
                            if not result:
                                continue
                            title_element = result.query_selector("a.fw-bold")
                            if not title_element:
                                continue

                            title = title_element.inner_text()
                            link = title_element.get_attribute("href")
                            url = f"https://ieeexplore.ieee.org{link}"

                            author_element = result.query_selector(".text-base-md-lh")
                            authors = author_element.inner_text().replace("\n", " ").strip() if author_element else "Unknown"

                            journal_element = result.query_selector(".fw-bold")
                            journal = journal_element.inner_text() if journal_element else "Unknown"

                            year_element = result.query_selector(".publisher-info-container")
                            if year_element:
                                year_text = year_element.inner_text()
                                match = re.search(r'\b\d{4}\b', year_text)
                                year = match.group(0) if match else "Unknown"
                            else:
                                year = "Unknown"

                            abstract_element = result.query_selector(".twist-container")
                            abstract = abstract_element.inner_text() if abstract_element else "Unknown"

                            # Escribir en formato BibTeX
                            file.write(f"@article{{ref{current_page}_{i},\n")
                            file.write(f"  title = {{{title}}},\n")
                            file.write(f"  author = {{{authors}}},\n")
                            file.write(f"  year = {{{year}}},\n")
                            file.write(f"  journal = {{{journal}}},\n")
                            file.write(f"  abstract = {{{abstract}}},\n")
                            file.write(f"  url = {{{url}}}\n")
                            file.write("}\n\n")

                        except Exception as e:
                            print(f"Error al procesar un resultado: {e}")

                    # Intentar ir a la siguiente página
                    try:
                        if current_page == 10:
                            print("Cargando las siguientes 10 páginas...")
                            next_button = page.locator('li.next-page-set button:has-text("Next")')
                            if next_button.is_visible():
                                next_button.click()
                                page.wait_for_timeout(5000)
                            else:
                                print("El botón 'Next' no está disponible.")
                                break
                        else:
                            print(f"Intentando ir a la página {current_page + 1}...")
                            next_page_button = page.locator(f'li button.stats-Pagination_{current_page + 1}')
                            if next_page_button.is_visible():
                                next_page_button.click()
                                page.wait_for_timeout(5000)
                            else:
                                print("Ya se alcanzó el límite.")
                                break

                        current_page += 1
                    except Exception as e:
                        print(f"No se pudo ir a la página {current_page + 1}: {e}")
                        break

                print(f"Los artículos se guardaron exitosamente en {filepath}")

        except Exception as e:
            print(f"Error general: {e}")
        finally:
            print("Los artículos de la base IEEE se guardaron exitosamente")
            browser.close()
            end_time = time.time()
            print(f"Scraper para Springer finalizado en {end_time - start_time:.2f} segundos.\n")

scrape_ieee()
