"""
LESSON 5: Tokyo Data Scout
Generate real Shibuya delivery scenario
"""

import numpy as np
import osmnx as ox
import networkx as nx

print("="*70)
print("FUJITSU CHALLENGE - TOKYO DATA GENERATION")
print("="*70)

# Shibuya Station coordinates
DEPOT = (35.6595, 139.7004)  # 渋谷駅

# 4 delivery stops (convenience stores, pharmacies, etc.)
STOPS = [
    (35.6612, 139.7021),  # Near Shibuya Crossing
    (35.6578, 139.6987),  # Near Mark City
    (35.6634, 139.7056),  # Near Yoyogi Park entrance
    (35.6556, 139.7034),  # Near Shibuya Stream
]

DEMANDS = [5, 3, 4, 3]  # kg
TIME_WINDOWS = [
    (9, 12),   # Morning delivery
    (10, 13),  # Late morning
    (14, 17),  # Afternoon
    (15, 18),  # Late afternoon
]

print(f"Depot: Shibuya Station {DEPOT}")
print(f"Stops: {len(STOPS)} locations")
for i, (coord, demand, tw) in enumerate(zip(STOPS, DEMANDS, TIME_WINDOWS)):
    print(f"  Stop {chr(65+i)}: {coord}, demand={demand}kg, time={tw}")

# Download street network (cached)
print("\nDownloading Tokyo street network...")
G = ox.graph_from_point(DEPOT, dist=2000, network_type='drive', simplify=True)

# Find nearest nodes
depot_node = ox.nearest_nodes(G, DEPOT[1], DEPOT[0])
stop_nodes = [ox.nearest_nodes(G, lon, lat) for lat, lon in STOPS]

print(f"\nNetwork: {len(G.nodes)} nodes, {len(G.edges)} edges")

# Calculate distance matrix (meters)
print("Calculating shortest paths...")
n = len(STOPS) + 1  # +1 for depot
dist_matrix = np.zeros((n, n))

all_nodes = [depot_node] + stop_nodes

for i in range(n):
    for j in range(n):
        if i == j:
            dist_matrix[i][j] = 0
        else:
            try:
                dist = nx.shortest_path_length(G, all_nodes[i], all_nodes[j], weight='length')
                dist_matrix[i][j] = dist  # meters
            except nx.NetworkXNoPath:
                dist_matrix[i][j] = 999999  # No path

print(f"\nDistance Matrix (meters):")
print(f"{'':>10}", end="")
for i in range(n):
    print(f"{i:>10}", end="")
print()
for i in range(n):
    print(f"{i:>10}", end="")
    for j in range(n):
        print(f"{dist_matrix[i][j]:>10.0f}", end="")
    print()

# Save data
np.save('tokyo_distances.npy', dist_matrix)
np.save('tokyo_demands.npy', np.array(DEMANDS))
np.save('tokyo_timewindows.npy', np.array(TIME_WINDOWS))

print("\n✓ Saved: tokyo_distances.npy, tokyo_demands.npy, tokyo_timewindows.npy")

# Visualize
import folium

m = folium.Map(location=DEPOT, zoom_start=16)
folium.Marker(DEPOT, popup="Depot (Shibuya Station)", icon=folium.Icon(color='red')).add_to(m)
for i, (coord, demand) in enumerate(zip(STOPS, DEMANDS)):
    folium.Marker(coord, popup=f"Stop {chr(65+i)} ({demand}kg)", 
                  icon=folium.Icon(color='blue')).add_to(m)

m.save('tokyo_route_map.html')
print("✓ Saved: tokyo_route_map.html")