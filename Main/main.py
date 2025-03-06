from Scrape.Scrape_IEE import scrape_ieee
from Scrape.Scrape_SAGE import scrape_sage
from Scrape.Scrape_ACM import scrape_acm
from Scrape.Unificar import unify_results

if __name__ == "__main__":
    ieee_data = scrape_ieee()
    sage_data = scrape_sage()
    acm_data = scrape_acm()

    unify_results(ieee_data, sage_data, acm_data)
    print("Datos unificados y duplicados almacenados correctamente.")
