import pandas as pd
import re
from tqdm import tqdm
import unicodedata

def parse_large_bib(file_path):
    """Analiza archivos BibTeX grandes con manejo mejorado de campos y tipos"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    entry_pattern = re.compile(
        r'@(?P<type>\w+)\s*{\s*(?P<id>[^,\s]+)\s*,\s*'
        r'(?P<fields>(?:[^@]*?)\s*(?=\s*@\w+\s*{|$))',
        re.DOTALL
    )
    
    field_pattern = re.compile(
        r'(?P<key>\w+)\s*=\s*'
        r'(?P<value>\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})+\}|"[^"]*"|[^,\n]+)\s*,?\s*',
        re.DOTALL
    )
    
    entries = []
    for match in tqdm(entry_pattern.finditer(content), desc="Analizando entradas"):
        entry = {
            'ENTRYTYPE': match.group('type').lower(),
            'ID': match.group('id').strip()
        }
        
        fields_content = match.group('fields')
        for field in field_pattern.finditer(fields_content):
            key = field.group('key').lower()
            value = field.group('value').strip()
            
            if value.startswith('{') and value.endswith('}'):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            
            value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('utf-8')
            entry[key] = value.strip()
        
        entries.append(entry)
    
    return entries

def normalize_product_type(tipo):
    """
    Normaliza los tipos a 4 categorías principales:
    - Articulo
    - Conference Paper
    - Libro
    - Capitulo de libro
    """
    if not tipo or pd.isna(tipo):
        return 'Otro'
    
    tipo = str(tipo).lower().strip()
    
    # Mapeo de tipos a categorías principales
    article_keywords = ['article', 'research', 'review', 'original', 'empirical', 
                       'methodology', 'survey', 'case study', 'commentary', 'editorial',
                       'letter', 'abstract', 'perspective', 'educational', 'narrative',
                       'technical', 'systematic', 'application', 'opinion', 'essay', 'comment',
                       'rapid communication', 'mini-review', 'research paper', 'invited review']
    
    conference_keywords = ['conference', 'paper','proceeding', 'meeting', 'workshop', 'symposium',
                          'inproceeding', 'event', 'presentation', 'talk', 'poster', 'young', 'special', 'announcement',
                          'dossier', 'award']
    
    book_keywords = ['book', 'monograph', 'treatise', 'compendium']
    
    chapter_keywords = ['chapter', 'capitulo', 'sección', 'section', 'parte', 'summary']
    
    # Determinar la categoría principal
    for keyword in article_keywords:
        if keyword in tipo:
            return 'Articulo'
    
    for keyword in conference_keywords:
        if keyword in tipo:
            return 'Conference Paper'
    
    for keyword in book_keywords:
        if keyword in tipo:
            return 'Libro'
    
    for keyword in chapter_keywords:
        if keyword in tipo:
            return 'Capitulo de libro'
    
    return 'Otro'

def normalize_authors(author_str):
    """Normalización mejorada de autores excluyendo 'View all'"""
    if not author_str or pd.isna(author_str):
        return []
    
    clean_str = re.sub(r'[\{\}"\\]', '', str(author_str))
    clean_str = re.sub(r'\s+', ' ', clean_str).strip()
    
    # Excluir "View all" desde el principio si es el único autor
    if clean_str.lower() == 'view all':
        return []
    
    if ' and ' in clean_str.lower():
        authors = re.split(r'\s+and\s+', clean_str, flags=re.IGNORECASE)
    elif ',' in clean_str:
        authors = re.split(r',\s*(?=[A-Z][a-z])', clean_str)
    else:
        authors = [clean_str]
    
    normalized = []
    for author in authors:
        author = author.strip()
        # Excluir cualquier variación de "View all"
        if re.fullmatch(r'view\s+all', author.lower()):
            continue
        if ',' in author:
            parts = [p.strip() for p in author.split(',', 1)]
            if len(parts) == 2:
                normalized.append(f"{parts[1]} {parts[0]}")
            else:
                normalized.append(parts[0])
        elif author:
            normalized.append(author)
    
    return normalized

def clean_journal_name(name):
    """Limpia nombres de revistas"""
    if not name or pd.isna(name):
        return None
    
    name = re.sub(r'[^\w\s-]', '', str(name))
    name = re.sub(r'\s+', ' ', name).strip()
    return name.title()