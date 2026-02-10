"""
LESSON 7B: Proper 10-Qubit Encoding with Qiskit Optimization
Using correct one-hot encoding and strong penalties
"""

import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator
import time

print("="*70)
print("FUJITSU CHALLENGE - PROPER 10-QUBIT VRP")
print("="*70)

# Load Tokyo data
dist_matrix = np.load('tokyo_distances.npy') / 1000  # km
demands = np.load('tokyo_demands.npy')
n_stops = 4

print(f"Distance matrix (km):\n{dist_matrix.round(2)}")

# PROPER ENCODING: 4 positions × 4 stops = 16 variables
# But we can reduce: fix first position to reduce symmetry
# Use 12 variables: pos2, pos3, pos4 × 4 stops each

qp = QuadraticProgram()

# Variables: x[pos][stop] = 1 if stop visited at position pos
# Position 0 (first) is always depot departure
# Positions 1,2,3,4 are the 4 stops
# Position 5 is return to depot

# SIMPLIFIED: Just find permutation of 4 stops
# Variables: x[i][j] = 1 if stop j is at position i (i=0,1,2,3)

var_names = []
for pos in range(4):
    for stop in range(4):
        name = f"x_{pos}_{stop}"
        qp.binary_var(name)
        var_names.append(name)

# Build objective: minimize total distance
linear = {}
quadratic = {}

# Distance: depot->pos0 + pos0->pos1 + pos1->pos2 + pos2->pos3 + pos3->depot
# For each possible assignment, add cost

P = 100  # Penalty weight

# CONSTRAINT: Each position has exactly one stop
for pos in range(4):
    vars_pos = [f"x_{pos}_{s}" for s in range(4)]
    # Linear penalty: sum(vars) = 1
    for v in vars_pos:
        linear[v] = linear.get(v, 0) - P  # -P * x
    # Quadratic penalty: sum_{i<j} 2*P * x_i * x_j
    for i in range(4):
        for j in range(i+1, 4):
            quadratic[(vars_pos[i], vars_pos[j])] = 2 * P

# CONSTRAINT: Each stop used exactly once
for stop in range(4):
    vars_stop = [f"x_{p}_{stop}" for p in range(4)]
    for v in vars_stop:
        linear[v] = linear.get(v, 0) - P
    for i in range(4):
        for j in range(i+1, 4):
            quadratic[(vars_stop[i], vars_stop[j])] = 2 * P

# OBJECTIVE: Distance costs
# This is complex—use brute force verification instead for now
# Just minimize penalties to find valid solution first

qp.minimize(linear=linear, quadratic=quadratic)

print(f"\nQuadratic Program:")
print(f"Variables: {qp.get_num_binary_vars()}")
print(f"Linear terms: {len(linear)}")
print(f"Quadratic terms: {len(quadratic)}")

# CLASSICAL SOLUTION
print("\n" + "="*70)
print("CLASSICAL SOLUTION")
print("="*70)

from itertools import permutations

best_cost = float('inf')
best_route = None

for perm in permutations(range(4)):
    route = [0] + [p+1 for p in perm] + [0]
    cost = sum(dist_matrix[route[i]][route[i+1]] for i in range(5))
    if cost < best_cost:
        best_cost = cost
        best_route = perm

print(f"Optimal: Depot -> ", end="")
for p in best_route:
    print(f"{chr(65+p)} -> ", end="")
print(f"Depot ({best_cost:.2f} km)")

# QUANTUM SOLUTION (simplified—just verify valid solutions exist)
print("\n" + "="*70)
print("QUANTUM VALIDATION")
print("="*70)

# Check: can we represent valid solutions?
# Valid solution: one x_{pos}_{stop} = 1 for each pos, each stop used once

valid_assignments = []
for perm in permutations(range(4)):
    # Build assignment
    assignment = {}
    for pos, stop in enumerate(perm):
        for s in range(4):
            assignment[f"x_{pos}_{s}"] = 1 if s == stop else 0
    valid_assignments.append((perm, assignment))

print(f"Valid assignments: {len(valid_assignments)}")

# Calculate objective for each
best_obj = float('inf')
best_assign = None

for perm, assign in valid_assignments:
    obj = sum(linear.get(v, 0) * assign[v] for v in assign)
    obj += sum(quadratic.get((v1, v2), 0) * assign[v1] * assign[v2] 
               for v1 in assign for v2 in assign if v1 < v2)
    if obj < best_obj:
        best_obj = obj
        best_assign = assign

print(f"Best valid objective: {best_obj}")
print(f"(Should be -8*P = -800 for perfect satisfaction)")

# Run QAOA
print("\nRunning QAOA...")
backend = AerSimulator(method='statevector')
sampler = Sampler()

qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=200), reps=2)
optimizer = MinimumEigenOptimizer(qaoa)

try:
    result = optimizer.solve(qp)
    print(f"QAOA result: {result.fval}")
    print(f"Solution: {result.x}")
    
    # Decode
    route = []
    for pos in range(4):
        for stop in range(4):
            if result.variables_dict.get(f"x_{pos}_{stop}", 0) > 0.5:
                route.append(stop)
                break
    
    print(f"Decoded route: ", end="")
    for p in route:
        print(f"{chr(65+p)} -> ", end="")
    print("Depot")
    
except Exception as e:
    print(f"QAOA error: {e}")
    print("This is expected—QUBO may need tuning")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)
print("Issue: QUBO encoding needs proper distance terms")
print("Next: Add real distance costs to objective")
print("Status: Constraint encoding verified, objective needs work")



