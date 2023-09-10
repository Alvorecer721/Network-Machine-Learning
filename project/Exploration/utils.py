import pandas as pd
import random
import networkx as nx
import os
from collections import Counter, OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from scipy.stats import poisson

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



# Reproducibility
random.seed(42)  # for networkx
np.random.seed(42)  # for numpy

# Dictionary mapping route types to their names
ROUTE_TYPE_NAMES = {
    0: 'Tram',
    1: 'Subway',
    2: 'Rail',
    3: 'Bus',
    4: 'Ferry',
    5: 'Cable car',
    6: 'Gondola',
    7: 'Funicular'
}


def check_duplicate_route(cities, mode, path_prefix="../Data/all_cities"):
    duplicate_cities = []

    for city in cities:
        edges_path = os.path.join(path_prefix, city, f"network_{mode}.csv")
        edges = pd.read_csv(edges_path, sep=";")
        grouped_edges = edges.groupby(['from_stop_I', 'to_stop_I']).size()

        # Check if there are any duplicate routes
        if any(grouped_edges > 1):
            duplicate_cities.append(city)

    return duplicate_cities


def load_graph(city, mode, files_prefix="../Data/all_cities"):
    edges_path = os.path.join(files_prefix, city, f"network_{mode}.csv")
    nodes_path = os.path.join(files_prefix, city, "network_nodes.csv")

    edges = pd.read_csv(edges_path, sep=";")
    nodes = pd.read_csv(nodes_path, sep=";")

    graph = nx.MultiDiGraph()

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
            graph.add_edge(
                row.from_stop_I,
                row.to_stop_I,
                d=row.d,
                d_walk=row.d_walk,
                route_type='Walk'
            )
            # Add the edge in the opposite direction for 'walk' mode
            graph.add_edge(
                row.to_stop_I,
                row.from_stop_I,
                d=row.d,
                d_walk=row.d_walk,
                route_type='Walk'
            )
        else:
            graph.add_edge(
                row.from_stop_I,
                row.to_stop_I,
                d=row.d,
                duration_avg=row.duration_avg,
                n_vehicles=row.n_vehicles,
                route_type=ROUTE_TYPE_NAMES[row.route_type] if 'route_type' in edges.columns else None
            )

    return graph


def vis_graph(graph, ax, node_size=0.5, no_edge_alpha=0.5):
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


def list_cities(root_path="../data/all_cities"):
    sub = os.listdir(root_path)
    sub = [x for x in sub if os.path.isdir(os.path.join(root_path, x))]
    return sub


def plot_route_percentages(city, G=None, ax=None):
    # Load the city graph
    if G is None:
        G = load_graph(city, 'combined')

    # Get a list of all the route types for each edge in the graph
    route_types = [d.get('route_type') for _, _, d in G.edges(data=True) if 'route_type' in d]

    # Count the occurrence of each route type
    route_counts = Counter(route_types)
    route_counts = {key: count for key, count in route_counts.items() if
                    count > 0}  # only show existing transportation mode

    # Compute the total number of edges
    total_edges = G.number_of_edges()

    # Convert counts to percentages
    # route_percentages = {route_type: count / total_edges * 100 for route_type, count in route_counts.items()}
    route_percentages = {route_type: route_counts[route_type] / total_edges * 100 for route_type in
                         route_counts.keys()}

    # Sort in descending order
    route_percentages = OrderedDict(sorted(route_percentages.items(), key=lambda x: x[1], reverse=True))

    # Check if an axis object is provided
    if ax is None:
        fig, ax = plt.subplots()

    # Plot
    rects = ax.bar(route_percentages.keys(), route_percentages.values())
    ax.set_title(f"{city.capitalize()}")
    ax.set_xticks(range(len(route_percentages)))
    ax.set_xticklabels(route_percentages.keys(), rotation=45, horizontalalignment='right')

    # Add exact statistics on each bar
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.005 * height,
                '{:.2f}%'.format(height),
                ha='center', va='bottom')


def plot_degree_distribution(city, x_scale="log", y_scale="log", G=None, ax=None):
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


def inspect_distributions(Gs, cities, distribution, mu=2, C=10, gamma=4):
    """
    Plot degree distributions for a list of graphs and annotate them with city names.

    Parameters:
    - Gs: List of networkx.Graph objects
    - cities: List of city names
    - distribution: Type of distribution to compare against ('power-law' or 'poisson')
    - mu: Mean for the Poisson distribution
    - C, gamma: Parameters for the power law distribution

    Returns: Nothing, but produces a plot.
    """
    # Assert that Gs and cities have the same length
    assert len(Gs) == len(cities), "Each graph should have a corresponding city name."

    # Calculate max degree for setting x-axis limit
    max_degree = max([max(g.degree, key=lambda x: x[1])[1] for g in Gs])

    # Create a subplot grid
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))

    # Plot degree distributions for each city
    for ax, city, graph in zip(axs.flatten(), cities, Gs):
        ax.set_xlim(1, max_degree)
        plot_degree_distribution(
            city=city,
            G=graph,
            ax=ax,
            x_scale="log",
            y_scale="log"
        )

        # Choose the right comparison distribution
        x = np.linspace(1, max_degree, 100)
        if distribution == 'power-law':
            y = C * (x ** -gamma)
        elif distribution == 'poisson':
            y = poisson_density(x, mu)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Add the comparison distribution to the plot
        ax.plot(x, y)

    plt.tight_layout()
    plt.show()


def get_largest_scc(graph: nx.DiGraph):
    gccs = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    return graph.subgraph(gccs[0])


def expected_gcc_size(avg_degree, n_nodes):
    if avg_degree < 1:
        return np.log(n_nodes)
    elif 1 <= avg_degree <= np.log(n_nodes):
        return n_nodes ** (2 / 3)
    else:
        raise ValueError("Help")
        return (avg_degree / (n_nodes) - 1) * n_nodes


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

if __name__ == "__main__":
    load_graph('adelaide', 'walk')
