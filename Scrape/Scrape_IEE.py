from playwright.sync_api import sync_playwright
import os
import re
import time

# -------------------------------------------------------------
# USO DE CHATGPT PARA LA ESTRUCTURA DE LOS SCRAPES
# -------------------------------------------------------------

def scrape_ieee():

    # Crear la carpeta "Data" si no existe
    if not os.path.exists("Data"):
        os.makedirs("Data")

    with sync_playwright() as p:
        start_time = time.time()
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        try:

            # Paso 1: Acceder a la página principal
            page.goto("https://library.uniquindio.edu.co/databases")
            page.wait_for_load_state("domcontentloaded")

            # Paso 2: Hacer clic en "Fac. Ingeniería"
            fac_ingenieria_selector = "div[data-content-listing-item='fac-ingenier-a']"
            page.click(fac_ingenieria_selector)
            page.wait_for_load_state("domcontentloaded")

            # Paso 3: Hacer clic en "IEEXPLORER) "
            elements = page.locator("//a[contains(@href, \"https://ieeexplore-ieee-org.crai.referencistas.com/Xplore/home.jsp\")]//span[contains(text(), \"IEEE (Institute of Electrical and Electronics Engineers) - (DESCUBRIDOR)\")]")
            count = elements.count()

            for i in range(count):
                if elements.nth(i).is_visible():
                    elements.nth(i).click()
                    page.wait_for_load_state("domcontentloaded")
                    print(f"Se hizo clic en el elemento {i+1}")
                    break
            else:
                print("No se encontró un elemento visible con el texto deseado.")

            # Paso 4: Hacer clic en el botón de iniciar sesión con Google
            google_login_button = "a#btn-google"
            page.click(google_login_button)

            # Paso 5: Ingresar el correo electrónico
            email_input_selector = "input#identifierId"
            page.fill(email_input_selector, "jhojanr")
            next_button_selector = "button:has-text('Siguiente')"
            page.click(next_button_selector)
            page.wait_for_load_state("domcontentloaded")

            # Paso 6: Ingresar la contraseña
            password_input_selector = "input[name='Passwd']"
            page.fill(password_input_selector, "conras")
            page.click(next_button_selector)
            page.wait_for_load_state("domcontentloaded")
            print("Login exitoso, listo para comenzar el scraping.")

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
                MAX_PAGE = 45

                while current_page <= MAX_PAGE:  # Iterar hasta el límite de página
                    print(f"Procesando página {current_page}...")

                    # Procesar los resultados actuales
                    page.wait_for_selector(".List-results-items", timeout=60000)
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

                            journal_element = result.query_selector("div.description > a[xplhighlight]")
                            journal = journal_element.inner_text() if journal_element else "Unknown"

                            year_element = result.query_selector(".publisher-info-container")
                            if year_element:
                                year_text = year_element.inner_text()
                                match = re.search(r'\b\d{4}\b', year_text)
                                year = match.group(0) if match else "Unknown"
                            else:
                                year = "Unknown"

                            tipo_elements = result.query_selector_all("span[xplhighlight]")
                            tipo = "Unknown"

                            for element in tipo_elements:
                                text = element.inner_text().strip()
                                # Excluir si contiene "Year:", "Volume:", "Issue:" O cualquier dígito (0-9)
                                if not (
                                    "Year:" in text 
                                    or "Volume:" in text 
                                    or "Issue:" in text 
                                    or re.search(r'\d', text)  # Busca cualquier número
                                ):
                                    tipo = text
                                    break

                            # PUBLISHER
                            publisher_element = result.query_selector("button[xplhighlight] span.title")  # Localiza el título "Publisher:"
                            if publisher_element and "Publisher:" in publisher_element.inner_text():
                                # Busca el siguiente span hermano que contiene el nombre (IEEE)
                                publisher = publisher_element.query_selector("xpath=following-sibling::span[1]").inner_text().strip()
                            else:
                                publisher = "Unknown"

                            abstract_element = result.query_selector(".twist-container")
                            abstract = abstract_element.inner_text() if abstract_element else "Unknown"

                            # Escribir en formato BibTeX
                            file.write(f"@article{{ref{current_page}_{i},\n")
                            file.write(f"  title = {{{title}}},\n")
                            file.write(f"  author = {{{authors}}},\n")
                            file.write(f"  year = {{{year}}},\n")
                            file.write(f"  journal = {{{journal}}},\n")
                            file.write(f"  tipo = {{{tipo}}},\n")
                            file.write(f"  publisher = {{{publisher}}},\n")
                            file.write(f"  abstract = {{{abstract}}},\n")
                            file.write(f"  url = {{{url}}}\n")
                            file.write("}\n\n")

                        except Exception as e:
                            print(f"Error al procesar un resultado: {e}")

                    # Intentar ir a la siguiente página
                    try:
                        if current_page in [10, 20, 30, 40]:
                            print("Cargando las siguientes 10 páginas...")
                            next_button = page.locator('li.next-page-set button:has-text("Next")')
                            if next_button.is_visible():
                                next_button.click()
                                page.wait_for_timeout(5000)
                                page.wait_for_selector(".List-results-items", timeout=60000)
                            else:
                                print("El botón 'Next' no está disponible.")
                                break
                        else:
                            print(f"Intentando ir a la página {current_page + 1}...")
                            next_page_button = page.locator(f'li button.stats-Pagination_{current_page + 1}')
                            if next_page_button.is_visible():
                                next_page_button.click()
                                page.wait_for_timeout(5000)
                                page.wait_for_selector(".List-results-items", timeout=60000)
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
            print(f"Scraper finalizado en {end_time - start_time:.2f} segundos.")

scrape_ieee()
