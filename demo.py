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
import dimod
import sys
import networkx as nx
import numpy as np
from dwave.system import LeapHybridSampler

import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

# Read in user option for random seed
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="set a random seed for scenario", type=int)
args = parser.parse_args()

if isinstance(args.seed, int):
    random.seed(args.seed)
elif args.seed is None:
    pass
else:
    print("Seed must be an integer.")
    sys.exit(0)

# Build large grid graph for city
G = nx.grid_2d_graph(14, 15)
nodes = G.nodes()

# Tunable parameters
gamma1 = len(G.nodes()) * 2
gamma2 = len(G.nodes()) / 3
gamma3 = len(G.nodes()) * 0.6
gamma4 = len(G.nodes()) ** 2

# Identify a fixed set of points of interest
num_poi = 3
pois = random.sample(nodes, k=num_poi)

# Identify a fixed set of current charging locations
num_cs = 4
charging_stations = random.sample(nodes, k=num_cs)

# Number of new charging stations to place
num_new_cs = 2

poi_cs_list = set(pois) - (set(pois)-set(charging_stations))
poi_cs_graph = G.subgraph(poi_cs_list)
poi_graph = G.subgraph(pois)
cs_graph = G.subgraph(charging_stations)

# Identify potential new charging locations
potential_new_cs_nodes = list(nodes - charging_stations)

# Build BQM using adjVectors to find best new charging location s.t. min 
# distance to POIs and max distance to existing charging locations
bqm = dimod.AdjVectorBQM(len(potential_new_cs_nodes), 'BINARY')

# Constraint 1: Min average distance to POIs
for i in range(len(potential_new_cs_nodes)):
    # Compute average distance to POIs from this node
    ave_dist = 0
    cand_loc = potential_new_cs_nodes[i]
    for loc in pois:
        dist = (cand_loc[0]**2 - 2*cand_loc[0]*loc[0] + loc[0]**2 
                            + cand_loc[1]**2 - 2*cand_loc[1]*loc[1] + loc[1]**2)
        ave_dist += dist / num_cs
    bqm.linear[i] = ave_dist * gamma1

# Constraint 2: Max distance to existing chargers
for i in range(len(potential_new_cs_nodes)):
    # Compute average distance to POIs from this node
    ave_dist = 0
    cand_loc = potential_new_cs_nodes[i]
    for loc in charging_stations:
        dist = (-1*cand_loc[0]**2 + 2*cand_loc[0]*loc[0] - loc[0]**2
                            - cand_loc[1]**2 + 2*cand_loc[1]*loc[1] - loc[1]**2)
        ave_dist += dist / num_poi
    bqm.linear[i] += ave_dist * gamma2

# Constraint 3: Max distance to other new charging location
for i in range(len(potential_new_cs_nodes)):
    for j in range(i+1, len(potential_new_cs_nodes)):
        ai = potential_new_cs_nodes[i]
        aj = potential_new_cs_nodes[j]
        dist = (-1*ai[0]**2 + 2*ai[0]*aj[0] - aj[0]**2 - ai[1]**2 
                + 2*ai[1]*aj[1] - aj[1]**2)
        bqm.add_interaction(i, j, dist * gamma3)

# Constraint 4: Choose exactly num_new_cs new charging locations
bqm.update(dimod.generators.combinations(bqm.variables, num_new_cs, strength=gamma4))

# Run BQM on HSS
sampler = LeapHybridSampler()
sampleset = sampler.sample(bqm)

# Process result and print information to command-line
print("\nSolution returned: \n------------------")
ss = sampleset.first.sample
new_charging_nodes = [potential_new_cs_nodes[k] for k, v in ss.items() if v == 1]
print("\nNew charging locations:\t\t\t\t", new_charging_nodes)

poi_ave_dist = [0,0]
for loc in pois:
    poi_ave_dist[0] += (1/num_poi)*(abs(new_charging_nodes[0][0]-loc[0])+abs(new_charging_nodes[0][1]-loc[1]))
    poi_ave_dist[1] += (1/num_poi)*(abs(new_charging_nodes[1][0]-loc[0])+abs(new_charging_nodes[1][1]-loc[1]))
print("Average distance to POIs:\t\t\t", poi_ave_dist)

old_cs_ave_dist = [0,0]
for loc in charging_stations:
    old_cs_ave_dist[0] += (1/num_cs)*(abs(new_charging_nodes[0][0]-loc[0])+abs(new_charging_nodes[0][1]-loc[1]))
    old_cs_ave_dist[1] += (1/num_cs)*(abs(new_charging_nodes[1][0]-loc[0])+abs(new_charging_nodes[1][1]-loc[1]))
print("Average distance to old charging stations:\t", old_cs_ave_dist)

new_cs_dist = abs(new_charging_nodes[0][0]-new_charging_nodes[1][0])+abs(new_charging_nodes[0][1]-new_charging_nodes[1][1])
print("Distance between new chargers:\t\t\t", new_cs_dist)

# Display image of results for user
#   - Black nodes: available space
#   - Red nodes: current charger location
#   - Nodes marked 'P': POI locations
#   - Blue nodes: new charger locations

new_cs_graph = G.subgraph(new_charging_nodes)
poi_labels = {x: 'P' for x in poi_graph.nodes()}
poi_cs_labels = {x: 'P' for x in poi_graph.nodes()}
pos = {x: [x[0],x[1]] for x in G.nodes()}

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('New EV Charger Locations')

# Draw existing set up (left image)
nx.draw_networkx(G, ax=ax1, pos=pos, with_labels=False, node_color='k', font_color='w')
nx.draw_networkx(poi_graph, ax=ax1, pos=pos, with_labels=True, 
                    labels=poi_labels, node_color='k', font_color='w')
nx.draw_networkx(cs_graph, ax=ax1, pos=pos, with_labels=False, node_color='r',
                    font_color='k')
nx.draw_networkx(poi_cs_graph, ax=ax1, pos=pos, with_labels=True, 
                    labels=poi_cs_labels, node_color='r', font_color='w')

# Draw new set up (right image)
nx.draw_networkx(G, ax=ax2, pos=pos, with_labels=False, node_color='k', 
                    font_color='w')
nx.draw_networkx(poi_graph, ax=ax2, pos=pos, with_labels=True, 
                    labels=poi_labels,node_color='k', font_color='w')
nx.draw_networkx(cs_graph, ax=ax2, pos=pos, with_labels=False, node_color='r',
                    font_color='k')
nx.draw_networkx(poi_cs_graph, ax=ax2, pos=pos, with_labels=True, 
                    labels=poi_cs_labels,  node_color='r', font_color='w')
nx.draw_networkx(new_cs_graph, ax=ax2, pos=pos, with_labels=False, 
                    node_color='#00b4d9', font_color='w')

# Save image
plt.savefig("map.png")
