#!/usr/bin/env python
# Copyright 2020 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from qdeepsdk import QDeepHybridSolver

def read_in_args():
    """Read user-specified parameters from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--width", help="grid width", default=15, type=int)
    parser.add_argument("-y", "--height", help="grid height", default=15, type=int)
    parser.add_argument("-p", "--poi", help="number of points of interest", default=3, type=int)
    parser.add_argument("-c", "--chargers", help="number of existing chargers", default=4, type=int)
    parser.add_argument("-n", "--new-chargers", help="number of new chargers", default=2, type=int)
    args = parser.parse_args()
    if args.poi > args.width * args.height or (args.chargers + args.new_chargers) > args.width * args.height:
        print("Grid size is not large enough for scenario.")
        exit(0)
    return args

def set_up_scenario(w, h, num_poi, num_cs):
    """Create a grid scenario with randomly chosen points of interest and existing chargers."""
    G = nx.grid_2d_graph(w, h)
    nodes = list(G.nodes)
    pois = random.sample(nodes, k=num_poi)
    charging_stations = random.sample(nodes, k=num_cs)
    potential_new_cs_nodes = list(set(nodes) - set(charging_stations))
    return G, pois, charging_stations, potential_new_cs_nodes

def distance(a, b):
    """Euclidean squared distance between two nodes."""
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def build_qubo_loops(potential_new_cs_nodes, num_poi, pois, num_cs, charging_stations, num_new_cs):
    """Build QUBO matrix using loops.
    
    Four constraints are applied:
      1. Minimize average distance to POIs.
      2. Maximize distance to existing chargers.
      3. Maximize separation between new chargers.
      4. Force exactly num_new_cs new charger selections.
    
    Returns:
        Q: a symmetric NumPy array representing the QUBO.
    """
    n = len(potential_new_cs_nodes)
    Q = np.zeros((n, n))
    
    # Constraint 1: Minimize average distance to POIs.
    gamma1 = n * 4.0
    for i in range(n):
        cand = potential_new_cs_nodes[i]
        avg_dist = sum(distance(cand, poi) for poi in pois) / num_poi
        Q[i, i] += avg_dist * gamma1

    # Constraint 2: Maximize average distance to existing chargers.
    gamma2 = n / 3.0
    for i in range(n):
        cand = potential_new_cs_nodes[i]
        avg_dist_cs = sum(distance(cand, cs) for cs in charging_stations) / num_cs
        Q[i, i] += -avg_dist_cs * gamma2

    # Constraint 3: Maximize distance between new charger locations.
    gamma3 = n * 1.7
    for i in range(n):
        for j in range(i+1, n):
            d = distance(potential_new_cs_nodes[i], potential_new_cs_nodes[j])
            Q[i, j] += -d * gamma3
            Q[j, i] = Q[i, j]

    # Constraint 4: Force exactly num_new_cs new charger nodes.
    gamma4 = n**3
    for i in range(n):
        Q[i, i] += gamma4 * (1 - 2*num_new_cs)
    for i in range(n):
        for j in range(i+1, n):
            Q[i, j] += 2 * gamma4
            Q[j, i] = Q[i, j]

    return Q

def run_qubo_solver(Q):
    """Solve the QUBO problem using QDeepHybridSolver."""
    solver = QDeepHybridSolver()
    # Set your authentication token (replace with a valid token)
    solver.token = "mtagdfsplb"
    response = solver.solve(Q)
    return response

def save_output_image(G, pois, charging_stations, new_charger_nodes):
    """Generate and save an image of the grid scenario."""
    fig, ax = plt.subplots(figsize=(8,8))
    pos = {node: (node[0], node[1]) for node in G.nodes()}
    nx.draw(G, pos=pos, node_color='lightgray', ax=ax)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=pois, node_color='black', label='POI')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=charging_stations, node_color='red', label='Existing Charger')
    nx.draw_networkx_nodes(G, pos=pos, nodelist=new_charger_nodes, node_color='#00b4d9', label='New Charger')
    ax.legend()
    plt.savefig("map.png")
    plt.close()

if __name__ == '__main__':
    args = read_in_args()
    G, pois, charging_stations, potential_new_cs_nodes = set_up_scenario(args.width, args.height, args.poi, args.chargers)
    Q = build_qubo_loops(potential_new_cs_nodes, args.poi, pois, args.chargers, charging_stations, args.new_chargers)
    response = run_qubo_solver(Q)
    solution = response.get('QdeepHybridSolver', {})
    config = solution.get('configuration', [])
    new_charger_nodes = [potential_new_cs_nodes[i] for i, bit in enumerate(config) if bit == 1]
    print("\nHybrid Solver Results:")
    print(solution)
    save_output_image(G, pois, charging_stations, new_charger_nodes)
