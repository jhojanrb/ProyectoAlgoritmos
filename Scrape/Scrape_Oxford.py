from playwright.sync_api import sync_playwright
import time
import os
import re

def scrape_oxford_with_university_login():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Ejecutar en modo no headless para depuración
        page = browser.new_page()
        start_time = time.time()

        try:
            # Paso 1: Acceder a la página principal
            page.goto("https://library.uniquindio.edu.co/databases")
            page.wait_for_load_state("domcontentloaded")

            # Paso 2: Hacer clic en "Fac. Ingeniería"
            fac_ingenieria_selector = "div[data-content-listing-item='fac-ingenier-a']"
            page.click(fac_ingenieria_selector)
            page.wait_for_load_state("domcontentloaded")

            # Paso 3: Hacer clic en "OXFORD Revistas Consorcio Colombia"
            elements = page.locator("//a[contains(@href, 'academic.oup.com/journals')]//span[contains(text(), 'OXFORD Revistas Consorcio Colombia')]")
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
            
            # Buscar artículos
            search_selector = "input[id='SitePageHeader-microsite-search-term']"
            page.wait_for_selector(search_selector, timeout=60000)
            page.fill(search_selector, "computational thinking")
            page.press(search_selector, "Enter")
            page.wait_for_selector(".search__item", timeout=60000)

            # Guardar resultados en un archivo BibTeX
            filepath = os.path.join("Data", "resultados_Oxford.bib")
            with open(filepath, mode="w", encoding="utf-8") as file:
                for page_num in range(1, 1001):  # Iterar hasta la página 1000
                    print(f"Procesando página {page_num}...")

                    # Revalidar que los resultados están disponibles
                    page.wait_for_selector(".search__item", timeout=60000)
                    results = page.query_selector_all(".search__item")

                    for i, result in enumerate(results):
                        try:
                            # Extraer información del artículo
                            title = result.query_selector(".access-title").inner_text() if result.query_selector(".hlFld-Title a") else "Unknown"
                            link = result.query_selector(".hlFld-Title a").get_attribute("href") if result.query_selector(".hlFld-Title a") else "Unknown"
                            authors = result.query_selector(".rlist--inline").inner_text() if result.query_selector(".rlist--inline") else "Unknown"

                            year_element = result.query_selector(".bookPubDate")
                            year = re.search(r'\b\d{4}\b', year_element.inner_text()).group(0) if year_element and re.search(r'\b\d{4}\b', year_element.inner_text()) else "Unknown"
                            journal = result.query_selector(".issue-item__detail").inner_text() if result.query_selector(".issue-item__detail") else "Unknown"
                            abstract = result.query_selector(".issue-item__abstract").inner_text() if result.query_selector(".issue-item__abstract") else "Unknown"

                            # Escribir en formato BibTeX
                            file.write(f"@article{{ref{page_num}_{i},\n")
                            file.write(f"  title = {{{title}}},\n")
                            file.write(f"  author = {{{authors}}},\n")
                            file.write(f"  year = {{{year}}},\n")
                            file.write(f"  journal = {{{journal}}},\n")
                            file.write(f"  abstract = {{{abstract}}},\n")
                            file.write(f"  url = {{{'https://dl.acm.org' + link}}}\n")
                            file.write("}\n\n")
                        except Exception as e:
                            print(f"Error al procesar un resultado en la página {page_num}: {e}")

                    # Avanzar a la siguiente página con reintentos
                    retries = 3
                    while retries > 0:
                        try:
                            next_button = page.query_selector(".pagination__btn--next")
                            if next_button:
                                next_button.click()
                                time.sleep(3)  # Esperar 3 segundos antes de cargar la siguiente página
                                page.wait_for_load_state("domcontentloaded", timeout=90000)  # Incrementar el tiempo de espera
                                break
                            else:
                                print("No se encontró el botón de siguiente. Finalizando.")
                                return
                        except Exception as e:
                            retries -= 1
                            print(f"Reintentando cargar la página {page_num + 1}. Intentos restantes: {retries}")
                            time.sleep(5)  # Pausa antes del siguiente intento
                    else:
                        print(f"Error al intentar cargar la página {page_num + 1}. Finalizando.")
                        break

            print(f"Los artículos se guardaron exitosamente en {filepath}")
        except Exception as e:
            print(f"Error general: {e}")
        finally:
            browser.close()
            end_time = time.time()
            print(f"Scraper para ACM finalizado en {end_time - start_time:.2f} segundos.\n")
        
        
# Llamar a la función
scrape_oxford_with_university_login()