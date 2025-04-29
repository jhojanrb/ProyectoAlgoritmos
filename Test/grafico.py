import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

def plot_cooccurrence_network_improved(abstracts, keywords_dict, output_path="cooccurrence_improved.png"):
    """Genera un gráfico de co-ocurrencia con diseño optimizado y barra de color asociada."""
    G = nx.Graph()
    
    # Construir el grafo
    for abstract in abstracts:
        present_keywords = set()
        for key, synonyms in keywords_dict.items():
            if any(synonym.lower() in abstract.lower() for synonym in synonyms):
                present_keywords.add(key)
        for k1 in present_keywords:
            for k2 in present_keywords:
                if k1 != k2:
                    if G.has_edge(k1, k2):
                        G[k1][k2]["weight"] += 1
                    else:
                        G.add_edge(k1, k2, weight=1)
    
    # Posiciones de los nodos utilizando Kamada-Kawai
    pos = nx.kamada_kawai_layout(G)
    
    # Tamaño de los nodos basado en el grado
    node_degrees = dict(G.degree())
    node_sizes = [v * 50 for v in node_degrees.values()]  # Escalado para mejor visualización
    
    # Pesos de las aristas
    edges = G.edges()
    edge_weights = [G[u][v]["weight"] for u, v in edges]
    
    # Configurar el rango de los pesos para la barra de color
    norm = Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Greys, norm=norm)
    sm.set_array([])  # Necesario para asociar la barra de color

    # Dibujar el grafo
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Dibujar nodos
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=list(node_degrees.values()), 
        cmap=plt.cm.plasma, 
        alpha=0.9,
        ax=ax
    )
    
    # Dibujar aristas
    nx.draw_networkx_edges(
        G, pos, 
        width=[w / 2 for w in edge_weights],  # Escalar pesos de las aristas
        edge_color=edge_weights, 
        edge_cmap=plt.cm.Greys, 
        edge_vmin=min(edge_weights),
        edge_vmax=max(edge_weights),
        alpha=0.8,
        ax=ax
    )
    
    # Dibujar etiquetas de los nodos
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10, 
        font_color="black", 
        font_weight="bold",
        ax=ax
    )
    
    # Configuración del título
    ax.set_title(
        "Keyword Co-occurrence Network", 
        fontdict={
            "fontsize": 20,
            "fontweight": "bold",
            "color": "darkblue"
        }
    )
    
    # Añadir la barra de color al eje principal
    cbar = fig.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Edge Weight", fontsize=12)
    
    # Guardar el gráfico
    plt.axis("off")
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

# Ejemplo de uso
abstracts = [
    "This study focuses on consumer behaviour and marketing strategy.",
    "Consumer satisfaction is key in brand loyalty and purchase intention.",
    "Qualitative research in ethnography explores customer perception.",
]
keywords_dict = {
    "consumer behaviour": ["consumer behaviour", "consumer behavior"],
    "marketing strategy": ["marketing strategy"],
    "customer satisfaction": ["customer satisfaction", "consumer satisfaction"],
    "brand loyalty": ["brand loyalty"],
    "purchase intention": ["purchase intention"],
    "ethnography": ["ethnography"],
    "qualitative research": ["qualitative research"],
}

plot_cooccurrence_network_improved(abstracts, keywords_dict, output_path="cooccurrence_improved.png")
