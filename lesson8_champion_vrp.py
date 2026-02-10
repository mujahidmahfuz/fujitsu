"""
LESSON 8: The Champion's Weapon
Complete 16-qubit Tokyo VRP with distances
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
from itertools import permutations
import time

print("="*70)
print("FUJITSU CHALLENGE - CHAMPION 16-QUBIT VRP")
print("="*70)

# Load data
dist_matrix = np.load('tokyo_distances.npy') / 1000
demands = np.load('tokyo_demands.npy')
n_stops = 4

# Build complete QUBO with DISTANCES
qp = QuadraticProgram()

# 16 variables: x[pos][stop]
for pos in range(4):
    for stop in range(4):
        qp.binary_var(f"x_{pos}_{stop}")

P = 100  # Penalty
BIG_M = 10  # Distance scaling

linear = {}
quadratic = {}

# CONSTRAINTS (same as before)
for pos in range(4):
    vars_pos = [f"x_{pos}_{s}" for s in range(4)]
    for v in vars_pos:
        linear[v] = linear.get(v, 0) - P
    for i in range(4):
        for j in range(i+1, 4):
            quadratic[(vars_pos[i], vars_pos[j])] = 2 * P

for stop in range(4):
    vars_stop = [f"x_{p}_{stop}" for p in range(4)]
    for v in vars_stop:
        linear[v] = linear.get(v, 0) - P
    for i in range(4):
        for j in range(i+1, 4):
            quadratic[(vars_stop[i], vars_stop[j])] = 2 * P

# DISTANCE OBJECTIVE
# For each assignment, add distance cost
# Depot->first + between stops + last->depot

# Pre-calculate all route costs
route_costs = {}
for perm in permutations(range(4)):
    route = [0] + [p+1 for p in perm] + [0]
    cost = sum(dist_matrix[route[i]][route[i+1]] for i in range(5))
    route_costs[perm] = cost

# Add to QUBO: if x_{pos}_{stop} = 1, add appropriate distance cost
# This is approximate—we add marginal costs

# Simplified: Add cost for each stop at each position
# Position 0 (first stop): cost = dist[depot][stop+1]
for stop in range(4):
    var = f"x_0_{stop}"
    linear[var] = linear.get(var, 0) + BIG_M * dist_matrix[0][stop+1]

# Position 3 (last stop): cost = dist[stop+1][depot]
for stop in range(4):
    var = f"x_3_{stop}"
    linear[var] = linear.get(var, 0) + BIG_M * dist_matrix[stop+1][0]

# Between positions: add quadratic costs for transitions
for pos in range(3):
    for s1 in range(4):
        for s2 in range(4):
            if s1 != s2:
                var1 = f"x_{pos}_{s1}"
                var2 = f"x_{pos+1}_{s2}"
                cost = BIG_M * dist_matrix[s1+1][s2+1]
                quadratic[(var1, var2)] = quadratic.get((var1, var2), 0) + cost

qp.minimize(linear=linear, quadratic=quadratic)

print(f"Variables: {qp.get_num_binary_vars()}")
print(f"Linear terms: {len([v for v in linear if linear[v] != 0])}")
print(f"Quadratic terms: {len(quadratic)}")

# CLASSICAL VERIFICATION
print("\n" + "="*70)
print("CLASSICAL OPTIMAL")
print("="*70)

best_cost = float('inf')
best_route = None
for perm in permutations(range(4)):
    route = [0] + [p+1 for p in perm] + [0]
    cost = sum(dist_matrix[route[i]][route[i+1]] for i in range(5))
    if cost < best_cost:
        best_cost = cost
        best_route = perm

print(f"Optimal: ", end="")
for p in best_route:
    print(f"{chr(65+p)}->", end="")
print(f"Depot = {best_cost:.2f} km")

# QUANTUM SOLUTION
print("\n" + "="*70)
print("QUANTUM SOLUTION (QAOA p=2)")
print("="*70)

backend = AerSimulator(method='statevector')
sampler = Sampler()

qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=300, rhobeg=0.5), reps=2)
optimizer = MinimumEigenOptimizer(qaoa)

print("Running QAOA (may take 30-60 seconds)...")
start = time.time()
result = optimizer.solve(qp)
qaoa_time = time.time() - start

print(f"QAOA energy: {result.fval:.2f}")
print(f"Time: {qaoa_time:.2f}s")

# Decode route
route = []
for pos in range(4):
    for stop in range(4):
        if result.variables_dict.get(f"x_{pos}_{stop}", 0) > 0.5:
            route.append(stop)
            break

print(f"Quantum route: ", end="")
for p in route:
    print(f"{chr(65+p)}->", end="")
print("Depot")

# Calculate true cost
if len(route) == 4 and len(set(route)) == 4:
    full_route = [0] + [r+1 for r in route] + [0]
    true_cost = sum(dist_matrix[full_route[i]][full_route[i+1]] for i in range(5))
    print(f"True distance: {true_cost:.2f} km")
    
    gap = abs(true_cost - best_cost)
    print(f"Gap to optimal: {gap:.2f} km ({gap/best_cost*100:.1f}%)")
    
    if gap < 0.1:
        print("✓ OPTIMAL: Quantum found best route!")
    elif gap < 0.5:
        print("✓ EXCELLENT: Near-optimal solution")
    else:
        print("~ GOOD: Valid solution found")
else:
    print("✗ INVALID: Route doesn't use all stops")

print("\n" + "="*70)
print("CHAMPION STATUS")
print("="*70)
print("16-qubit Tokyo VRP with distances: COMPLETE")
print("Next: Scale to 40 qubits for Fujitsu simulator")



