import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

def generate_and_save_charts(stats, output_folder):
    """Genera gráficos como imágenes PNG y los guarda en la carpeta especificada"""
    
    # Crear la carpeta si no existe
    os.makedirs(output_folder, exist_ok=True)
    
    # 1. Gráfico de Top Autores
    if 'top_15_autores' in stats:
        plt.figure(figsize=(12, 8))
        # Ordenar de mayor a menor y seleccionar los primeros 15
        top_authors = stats['top_15_autores'].sort_values(ascending=True).head(15)
        
        # Crear el gráfico de barras horizontales
        ax = top_authors.plot(kind='barh', color='coral')
        
        # Agregar etiquetas con el número correspondiente al final de cada barra
        for index, value in enumerate(top_authors):
            ax.text(value + 0.1, index, str(value), va='center')
        
        # Configuración del título y etiquetas
        plt.title('Top 15 Autores con más Publicaciones', pad=20)
        plt.xlabel('Número de Publicaciones')
        plt.ylabel('Autores')
        
        # Ajustar diseño, guardar imagen y cerrar figura
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'top15_autores.png'), dpi=300)
        plt.close()

    
    # 2. Gráfico de Distribución por Tipo
    if 'tipos_existentes' in stats:
        plt.figure(figsize=(10, 10))
        
        # Crear el gráfico de pastel usando plt.pie()
        wedges, texts = plt.pie(
            stats['tipos_existentes'].values,
            labels=None,  # No mostrar etiquetas directamente en el gráfico
            startangle=90,
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'],
            wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
        )
        
        # Crear leyenda separada
        labels = [f"{label}: {value} ({percentage:.1f}%)"
                for label, value, percentage in zip(
                    stats['tipos_existentes'].index,
                    stats['tipos_existentes'].values,
                    100 * stats['tipos_existentes'].values / stats['tipos_existentes'].sum()
                )]
        plt.legend(wedges, labels, title="Tipos de Publicación", loc="center left", bbox_to_anchor=(1, 0.5))
        
        plt.title('Distribución por Tipo de Publicación', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'distribucion_tipos.png'), dpi=300)
        plt.close()


    # 3. Gráfico de Evolución Temporal por Tipo
    if 'añoPublicacion_Portipo' in stats:
        plt.figure(figsize=(14, 8))
        stats['añoPublicacion_Portipo'].T.plot(
            kind='bar',
            stacked=True,
            figsize=(14, 8),
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        plt.title('Evolución de Publicaciones por Tipo', pad=20)
        plt.xlabel('Año')
        plt.ylabel('Número de Publicaciones')
        plt.legend(title='Tipo')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'evolucion_temporal.png'), dpi=300)
        plt.close()
    
    # 4. Gráfico de Top Journals por Tipo
    if 'top15_journals' in stats:
        # Primero obtener una copia de los datos sin la columna Total
        plot_data = stats['top15_journals'].drop(columns=['Total'], errors='ignore').copy()
        
        # Calcular la suma por fila (total de publicaciones por journal)
        plot_data['Sum'] = plot_data.sum(axis=1)
        
        # Ordenar por la suma total de mayor a menor
        plot_data = plot_data.sort_values('Sum', ascending=True).head(15)
        
        # Eliminar la columna Sum antes de graficar
        plot_data = plot_data.drop(columns=['Sum'])
        
        plt.figure(figsize=(14, 8))
        # Crear gráfico de barras horizontales apiladas
        ax = plot_data.plot(
            kind='barh',
            stacked=True,
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            width=0.8  # Ancho de las barras
        )
        
        plt.title('Top 15 Journals por Tipo de Publicación', pad=20, fontsize=14)
        plt.xlabel('Número de Publicaciones', fontsize=12)
        plt.ylabel('Journal', fontsize=12)
        
        # Mejorar la leyenda
        plt.legend(
            title='Tipo',
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            fontsize=10
        )
        
        plt.savefig(
            os.path.join(output_folder, 'top15_journals.png'),
            dpi=300,
            bbox_inches='tight'  # Para asegurar que la leyenda se incluya
        )
        plt.close()


    # 5. Gráfico de Top Publishers por Año
    if 'top15_publishers' in stats:
        # Excluir columna 'Total' para el gráfico
        plot_data = stats['top15_publishers'].drop(columns=['Total'], errors='ignore').head(15)
        
        plt.figure(figsize=(14, 8))
        plot_data.plot(
            kind='barh',
            stacked=True,
            figsize=(14, 8),
            cmap='viridis'
        )
        plt.title('Top 15 Publishers por Año', pad=20)
        plt.xlabel('Número de Publicaciones')
        plt.ylabel('Publisher')
        plt.legend(title='Año')
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'top15_publishers.png'), dpi=300)
        plt.close()