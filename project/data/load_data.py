import networkx as nx
import csv
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

ROOT_DIR = "all_cities"
city = "adelaide"
mode = "bus"

fields = {}


def load_network(city, mode):

    df = pd.read_csv("all_cities/adelaide/network_bus.csv", sep=";")
    edges = df[['from_stop_I', 'to_stop_I']]
    graph = nx.from_pandas_edgelist()

    breakpoint()


    with open(os.path.join(ROOT_DIR, city, f"network_{mode}.csv")) as f:
        reader = csv.reader(f, delimiter=";")
        """
        Skip header.
        Header fields: 
            from stop I
            to stop I
            d: straight-line distance
            average duration: travel time between stops (in seconds)
            n_vehicles: probably number of vehicles that went through the route over the whole data processing ?
        Ignored for now:
            route_I_counts: comma separated of route_id: number of operation (sums to n_vehicles)
            * I think route is like a bus/metro line, something like M1 metro from Renens to Flon
        """
        print(next(reader))
        for l in reader:
            from_idx = int(l[0])
            to_idx = int(l[1])
            d = int(l[2])
            avg_duration = float(l[3])
            n_vehicles = int(l[4])
            print(l)
    network = nx.N


    return None


aaa = load_network(city, mode)
breakpoint()


