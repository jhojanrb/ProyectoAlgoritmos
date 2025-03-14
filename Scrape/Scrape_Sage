from playwright.sync_api import sync_playwright
import os
import re
import time

def scrape_sage():
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

            # Paso 3: Hacer clic en "SAGE Revistas Consorcio Colombia - (DESCUBRIDOR) "
            elements = page.locator("//a[contains(@href, 'journals.sagepub.com')]//span[contains(text(), 'SAGE Revistas Consorcio Colombia - (DESCUBRIDOR) ')]")
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
            page.fill(email_input_selector, "jhojanr.ramirezb@uqvirtual.edu.co")
            next_button_selector = "button:has-text('Siguiente')"
            page.click(next_button_selector)
            page.wait_for_load_state("domcontentloaded")

            # Paso 6: Ingresar la contraseña
            password_input_selector = "input[name='Passwd']"
            page.fill(password_input_selector, "zenitsu1099682")
            page.click(next_button_selector)
            page.wait_for_load_state("domcontentloaded")
            print("Login exitoso, listo para comenzar el scraping.")

            # Espera y selecciona el botón de aceptar cookies
            try:
                # Verificar si el botón está dentro de un iframe
                frames = page.frames
                button_found = False

                for frame in frames:
                    # Buscar el botón en cada iframe
                    button = frame.query_selector("button#onetrust-accept-btn-handler")
                    if button:
                        button_found = True
                        button.click(force=True)
                        print("Cookies aceptadas desde iframe.")
                        break

                # Si no se encuentra en un iframe, buscar en la página principal
                if not button_found:
                    page.wait_for_selector("button#onetrust-accept-btn-handler", timeout=10000)
                    cookies_button = page.query_selector("button#onetrust-accept-btn-handler")
                    if cookies_button and cookies_button.is_visible():
                        cookies_button.click(force=True)
                        print("Cookies aceptadas.")
                    else:
                        print("El botón no es visible o no se puede interactuar con él.")

            except Exception as e:
             print(f"Error al intentar aceptar las cookies: {e}")

            # Buscar artículos
            search_selector = "input[name='AllField']"
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")

            # Espera y selecciona el botón de aceptar cookies
            try:
                # Verificar si el botón está dentro de un iframe
                frames = page.frames
                button_found = False

                for frame in frames:
                    # Buscar el botón en cada iframe
                    button = frame.query_selector("button#onetrust-accept-btn-handler")
                    if button:
                        button_found = True
                        button.click(force=True)
                        print("Cookies aceptadas desde iframe.")
                        break

                # Si no se encuentra en un iframe, buscar en la página principal
                if not button_found:
                    page.wait_for_selector("button#onetrust-accept-btn-handler", timeout=10000)
                    cookies_button = page.query_selector("button#onetrust-accept-btn-handler")
                    if cookies_button and cookies_button.is_visible():
                        cookies_button.click(force=True)
                        print("Cookies aceptadas.")
                    else:
                        print("El botón no es visible o no se puede interactuar con él.")

            except Exception as e:
                print(f"Error al intentar aceptar las cookies: {e}")
            
            # Esperar que los resultados se carguen
            page.wait_for_selector(".rlist.search-result__body.items-results > div", timeout=60000)
            results = page.query_selector_all(".rlist.search-result__body.items-results > div")
            print("articulos detectados")

            # Guardar resultados en un archivo BibTeX
            filepath = os.path.join("Data", "resultados_Sage.bib")
            with open(filepath, mode="w", encoding="utf-8") as file:
                for page_num in range(1, 5000):  # Iterar hasta la página 5000
                    print(f"Procesando página {page_num}...")

                    # Revalidar que los resultados están disponibles
                    page.wait_for_selector(".rlist.search-result__body.items-results > div", timeout=60000)
                    results = page.query_selector_all(".rlist.search-result__body.items-results > div")

                    for i, result in enumerate(results):
                        try:
                            # Extraer información del artículo
                            title = result.query_selector(".sage-search-title").inner_text() if result.query_selector(".sage-search-title") else "Unknown"
                            link = result.query_selector(".sage-search-title").get_attribute("href") if result.query_selector(".sage-search-title") else "Unknown"
                            authors = result.query_selector(".issue-item__authors").inner_text().replace("\n", ", ").strip() if result.query_selector(".issue-item__authors") else "Unknown"

                            year_element = result.query_selector(".issue-item__header")
                            year = re.search(r'\b\d{4}\b', year_element.inner_text()).group(0) if year_element and re.search(r'\b\d{4}\b', year_element.inner_text()) else "Unknown"
                            journal = result.query_selector(".issue-item__row").inner_text() if result.query_selector(".issue-item__row") else "Unknown"
                            abstract = result.query_selector(".issue-item__abstract__content").inner_text() if result.query_selector(".issue-item__abstract__content") else "Unknown"

                            # Escribir en formato BibTeX
                            file.write(f"@article{{ref{page_num}_{i},\n")
                            file.write(f"  title = {{{title}}},\n")
                            file.write(f"  author = {{{authors}}},\n")
                            file.write(f"  year = {{{year}}},\n")
                            file.write(f"  journal = {{{journal}}},\n")
                            file.write(f"  abstract = {{{abstract}}},\n")
                            file.write(f"  url = {{{'https://journals.sagepub.com' + link}}}\n")
                            file.write("}\n\n")
                        except Exception as e:
                            print(f"Error al procesar un resultado en la página {page_num}: {e}")

                    # Avanzar a la siguiente página con reintentos
                    retries = 3
                    while retries > 0:
                        try:
                            # Seleccionar el botón "Siguiente"
                            next_button = page.query_selector("a[aria-label='next']")
                            if next_button:
                                next_page_url = next_button.get_attribute("href")  # Captura la URL del siguiente enlace
                                if next_page_url:
                                    print(f"Navegando a la URL de la página {page_num + 1}")
                                    page.goto(next_page_url)  # Navegar directamente a la URL
                                    page.wait_for_selector(".rlist.search-result__body.items-results > div", timeout=60000)
                                    break
                                else:
                                    print("No se encontró el enlace 'href' en el botón 'Siguiente'. Finalizando.")
                                    return
                            else:
                                print("No se encontró el botón 'Siguiente'. Finalizando.")
                                return
                        except Exception as e:
                            retries -= 1
                            print(f"Error al cargar la página {page_num + 1}: {e}. Reintentando... Intentos restantes: {retries}")
                            time.sleep(5)
                    else:
                        print(f"No se pudo cargar la página {page_num + 1}. Finalizando.")
                        break


            print(f"Los artículos se guardaron exitosamente en {filepath}")
        except Exception as e:
            print(f"Error general: {e}")
        finally:
            browser.close()
            end_time = time.time()
            print(f"Scraper para Sage finalizado en {end_time - start_time:.2f} segundos.\n")

# Llamar a la función
scrape_sage()
