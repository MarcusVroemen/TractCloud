import networkx as nx
import numpy as np
from networkx.algorithms.community import greedy_modularity_communities

def small_worldness(G, L, C, niter=5):
    """
    Compute small-worldness of a graph using omega and sigma values.
    Omega compares the graph to a random graph, and sigma compares it to a regular lattice.
    """

    # L = nx.average_shortest_path_length(G, weight='weight')
    # C = nx.average_clustering(G, weight='weight')

    rand_graph = nx.random_reference(G, niter=niter)
    L_rand = nx.average_shortest_path_length(rand_graph, weight='weight')
    C_rand = nx.average_clustering(rand_graph, weight='weight')

    omega = L_rand / L - C / C_rand
    sigma = (C / C_rand) / (L / L_rand)
    return omega, sigma

def compute_graph_measures(connectome):
    """
    Compute characteristic path length, mean clustering coefficient, 
    mean normalized betweenness centrality, mean global efficiency, and mean local efficiency.
    
    Parameters:
    - connectome (np.ndarray): A 2D numpy array representing the adjacency matrix of the graph.
    
    Returns:
    - dict: A dictionary containing the computed metrics.
    """
    G = nx.from_numpy_array(connectome)  # Convert connectome to a NetworkX graph
    
    # Mean Clustering Coefficient (C)
    C = nx.average_clustering(G, weight='weight')
    
    # Characteristic Path Length (L) & Small-worldness (omega and sigma)
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G, weight='weight')
        # omega, sigma = small_worldness(G, L, C)
    else:
        L, omega, sigma = np.nan, np.nan, np.nan  # Handle case if the graph is disconnected


    # Mean Normalized Betweenness Centrality (B)
    B = np.mean(list(nx.betweenness_centrality(G, normalized=True, weight='weight').values()))

    # Mean Global Efficiency (E)
    E = nx.global_efficiency(G)

    # Mean Local Efficiency (Eloc)
    local_efficiencies = [nx.local_efficiency(G.subgraph(n)) for n in G.nodes()]
    Eloc = np.mean(local_efficiencies)

    # Degree: Average number of connections each node has
    degrees = dict(G.degree(weight=None))
    mean_degree = np.mean(list(degrees.values()))

    # Strength: Sum of the weights of the edges connected to each node
    strengths = dict(G.degree(weight='weight'))
    mean_strength = np.mean(list(strengths.values()))

    # Modularity (MOD)
    try:
        communities = greedy_modularity_communities(G, weight='weight')  # Detect communities
        mod = nx.algorithms.community.modularity(G, communities, weight='weight')  # Compute modularity
    except:
        mod = np.nan
    
    return {
        'Characteristic Path Length (L)': L,
        'Mean Clustering Coefficient (C)': C,
        # 'Small-worldness (Omega)': omega,
        # 'Small-worldness (Sigma)': sigma,
        'Mean Normalized Betweenness Centrality (B)': B,
        'Mean Global Efficiency (E)': E,
        'Mean Local Efficiency': Eloc,
        'Mean Degree': mean_degree,
        'Mean Strength': mean_strength,
        'Modularity (MOD)': mod
    }
