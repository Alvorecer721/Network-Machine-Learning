import pandas as pd
import networkx as nx
import os
from collections import Counter
import numpy as np
from scipy import special

from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# Dictionary mapping route types to their names

ALL_MODES = ['walk', 'tram', 'subway', 'rail', 'bus', 'ferry', 'cablecar', 'gondola', 'funicular', 'combined']

ROUTE_TYPE_NAMES = {
    -1: 'Walk',
    0: 'Tram',
    1: 'Subway',
    2: 'Rail',
    3: 'Bus',
    4: 'Ferry',
    # 5: 'Cable car',  # Ignored as only present in Prague and accounts for 0.06% of edges
    # 6: 'Gondola',  # Ignored as not not present in any city in dataset
    # 7: 'Funicular'  # Ignored as not not present in any city in dataset
}

ROUTE_NAME_TO_IDX = {name: idx for idx, name in ROUTE_TYPE_NAMES.items()}


def count_num_city_with_pt():
    all_cities = list_cities()

    mode_counts = []
    city_path = lambda c: os.path.join("..", "data", "all_cities", c, f"network_{mode}.csv")
    for mode in ALL_MODES:
        cities = [c for c in all_cities if os.path.exists(city_path(c))]
        mode_counts.append((mode, len(cities)))

    mode_counts.sort(key=lambda x: x[1], reverse=True)
    print("Number of cities with transportation mode:")
    for mode, count in mode_counts:
        print(f"\t{mode}: {count}")



def check_duplicate_route(cities, mode, path_prefix="../data/all_cities"):
    if mode not in ("walk", "combined", "bus"):
        # Not all cities have all modes of transportation; would crash with other values for 'mode'
        raise ValueError("Implemented for 'combined' and 'bus' modes only")
    duplicate_cities = []

    for city in cities:
        edges_path = os.path.join(path_prefix, city, f"network_{mode}.csv")
        edges = pd.read_csv(edges_path, sep=";")
        grouped_edges = edges.groupby(['from_stop_I', 'to_stop_I']).size()

        # Check if there are any duplicate routes
        if any(grouped_edges > 1):
            duplicate_cities.append(city)

    return duplicate_cities



def load_graph(city, mode, root_path="../data/all_cities", graph_type=nx.DiGraph):
    edges_path = os.path.join(root_path, city, f"network_{mode}.csv")
    nodes_path = os.path.join(root_path, city, "network_nodes.csv")

    edges = pd.read_csv(edges_path, sep=";")
    nodes = pd.read_csv(nodes_path, sep=";")

    graph = graph_type()
    for _, row in nodes.iterrows():
        graph.add_node(
            row.stop_I,
            lat=row.lat,
            lon=row.lon,
            name=row.name,
            pos=(row.lon, row.lat)
        )

    for _, row in edges.iterrows():
        if mode == 'walk':
            # Walk edges are undirected and have different attributes
            graph.add_edge(
                row.from_stop_I,
                row.to_stop_I,
                d=row.d,
                d_walk=row.d_walk,
                route_type=ROUTE_NAME_TO_IDX['Walk']
            )
            graph.add_edge(
                row.to_stop_I,
                row.from_stop_I,
                d=row.d,
                d_walk=row.d_walk,
                route_type=ROUTE_NAME_TO_IDX['Walk']
            )
        else:
            keys = {"d", "duration_avg", "n_vehicles"}
            if "route_type" in edges.columns:
                keys.add("route_type")
            graph.add_edge(
                row.from_stop_I,
                row.to_stop_I,
                **{k: v for k, v in {**row}.items() if k in keys}
            )

    return graph


def list_cities(root_path="../data/all_cities"):
    sub = os.listdir(root_path)
    sub = [x for x in sub if os.path.isdir(os.path.join(root_path, x))]
    return sub

def plot_route_percentages(city, G=None, ax=None, include_walk=False):
    if G is None:
        G = load_graph(city, 'combined')

    # Get a list of all the route types for each edge in the graph
    route_types = [d.get('route_type') for _, _, d in G.edges(data=True) if 'route_type' in d]

    # Count the occurrence of each route type
    route_counts = Counter(route_types)

    # Compute the total number of edges
    total_edges = G.number_of_edges()

    # Convert counts to percentages
    keys = list(ROUTE_TYPE_NAMES.keys())
    if not include_walk:
        keys.remove(-1)
    route_percentages = {route_type_idx: route_counts[route_type_idx] / total_edges * 100 for route_type_idx in keys}


    # Prepare data for plotting
    labels = [ROUTE_TYPE_NAMES[int(route_type)] for route_type in route_percentages.keys()]
    sizes = list(route_percentages.values())

    # Check if an axis object is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot
    rects = ax.bar(labels, sizes)
    ax.set_title(f"{city.capitalize()}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, horizontalalignment='right')

    # Add exact statistics on each bar
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,
                '{:.2f}%'.format(height),
                ha='center', va='bottom')

    # If no axis object was provided, show the plot
    # TODO: will never be None since you defined ax above
    if ax is None:
        plt.show()


def plot_degree_distribution(city, G=None, ax=None, x_scale="log", y_scale="log"):
    # Load the city graph
    if G is None:
        G = load_graph(city, 'combined')

    # Calculate the degree of each node
    degrees = [degree for _, degree in G.degree()]

    # Check if an axis object is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot the degree distribution
    ax.hist(degrees, bins=15, rwidth=0.5, density=True)
    ax.set_yscale(x_scale)
    ax.set_xscale(y_scale)
    ax.set_title(f"{city.capitalize()}")

    # If no axis object was provided, show the plot
    if ax is None:
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.title(f'Degree Distribution for {city.capitalize()}')
        plt.show()


def graph_average_degree(G):
    degrees = [degree for _, degree in G.degree()]
    return np.mean(degrees)


def get_largest_scc(graph: nx.DiGraph):
    gccs = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    return graph.subgraph(gccs[0])


def poisson_density(x, mu):
    """
    mu: <k> - the average degree of the nodes in the network
    """
    # Note: euler_gamma is expansion of factorial to complex numbers
    return np.exp(- mu) * (mu ** x) / special.gamma(x)


def compute_graph_main_statistics(graph: nx.DiGraph):
    n = graph.number_of_nodes()
    l = graph.number_of_edges()
    avg_in_degree = np.mean([deg for _, deg in graph.in_degree()])
    avg_out_degree = np.mean([deg for _, deg in graph.out_degree()])

    avg_in_deg2 = np.mean([deg ** 2 for _, deg in graph.in_degree()])
    avg_out_deg2 = np.mean([deg ** 2 for _, deg in graph.out_degree()])

    # TODO: maybe we should compoute the average shortest path length per cluster and then average (weighted by number of vertices in cluster ?)
    largest_scc = get_largest_scc(graph)
    largest_scc_size = largest_scc.number_of_nodes()
    avg_dist = nx.average_shortest_path_length(largest_scc)
    per_node_cc = nx.clustering(graph)
    clustering_coeff = np.mean([coeff for _, coeff in per_node_cc.items()])

    return {
        'n': n,
        'l': l,
        'avg_in_degree': avg_in_degree,
        'avg_out_degree': avg_out_degree,
        'avg_in_deg2': avg_in_deg2,
        'avg_out_deg2': avg_out_deg2,
        'avg_dist': avg_dist,
        'clustering_coeff': clustering_coeff,
        'giant_component_size': largest_scc_size,
    }


def symmetrize_adj(graph):
    adj = nx.adjacency_matrix(graph).todense()
    adj = np.array(adj)
    """
    If edge exists in one direction, wants to make it bidirectional; computing (adj + adj.T) / 2 would put 0.5 on unidirectional edges
    If edge already exists in both directions, just leave it as 1, don't want to have twos in the matrix
    """
    adj = adj + adj.T
    adj[adj > 1] = 1
    return adj


def remove_deg_zero_nodes(adjacency):
    deg = np.sum(adjacency, axis=1)
    to_keep = deg > 0
    return adjacency[to_keep, :][:, to_keep]


def compute_laplacian(adjacency: np.ndarray, normalize: str):
    """ normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    # Taken from assignment 3
    n = adjacency.shape[0]
    diag = np.sum(adjacency, axis=1)
    D = np.diag(diag)

    if normalize is None: # combinatorial Laplacian
        return D - adjacency 
    
    # avoid complex numbers according to the warning
    diag[diag == 0] = 1e-12 

    if normalize == 'sym': # symmetric normalized Laplacian
        # Degree matrix is diagonal, so this is equivalent to D^-0.5
        degree_matrix_inv_sqrt = np.diag(diag ** -0.5)
        L = np.eye(n) - degree_matrix_inv_sqrt @ adjacency @ degree_matrix_inv_sqrt
        L = (L + L.T) / 2  # Makes sure really symmetric
        return L
    
    if normalize == 'rw': # random walk Laplacian
        degree_matrix_inv = np.diag(diag ** -1)
        return np.eye(n) - degree_matrix_inv @ adjacency
    
    raise ValueError(f'Unknown normalization: {normalize}')


def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian33
        U (np.ndarray): corresponding eigenvectors.
    """
    # Your solution here ###########################################################
    # Check whether the matrix is symmetric
    if np.all(laplacian == laplacian.T):
        return np.linalg.eigh(laplacian)
    else:
        return np.linalg.eig(laplacian)


def viz_graph(graph, ax, node_size=0.5, no_edge_alpha=0.5):
    # Plot disconnected edges lighter
    degrees = nx.degree(graph)
    node_with_edges = []
    node_without_edges = []

    for node in graph.nodes():
        if degrees[node] > 0:
            node_with_edges.append(node)
        else:
            node_without_edges.append(node)

    # Split graph based on non-zero-degree
    no_edge_subgraph = graph.subgraph(node_without_edges).copy()
    with_edges_subgraph = graph.subgraph(node_with_edges).copy()

    # Get positions of nodes
    positions = nx.get_node_attributes(with_edges_subgraph, "pos")

    # Check and remove nodes without 'pos' attribute
    # TODO：this might be performed when loading the graph
    nodes_without_pos = [node for node in with_edges_subgraph.nodes() if node not in positions]
    with_edges_subgraph.remove_nodes_from(nodes_without_pos)

    # No edges
    nx.draw_networkx_nodes(
        no_edge_subgraph,
        pos=nx.get_node_attributes(no_edge_subgraph, "pos"),
        node_size=node_size,
        ax=ax,
        alpha=no_edge_alpha,
        node_color='lightgreen',
    )

    # With edges
    nx.draw_networkx(
        with_edges_subgraph,
        pos=nx.get_node_attributes(with_edges_subgraph, "pos"),
        with_labels=False,
        node_size=node_size,
        ax=ax,
        alpha=0.7,
        arrows=False,
        edge_color='orange',
    )