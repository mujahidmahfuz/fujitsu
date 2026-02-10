"""

Lesson 2: The QUBO Blacksmith
Manual Construction of 2-stop VRP QUBO

"""

import numpy as np

#Distance matrix
dist = np.array([
    [0, 5, 3],        # Depot(0), A(1), B(2)
    [5, 0, 4],
    [3, 4, 0]
])


# Binary Varibales: [x1, x2, x3, x4]
# x1 = A at positin 1, x2 = B at position 2
# x3 = A at postion 3, x4 = B at position 4

p = 20  # Penalty weight


# QUBO matrix Q (4x4 symmetric)
# Q [i][j] is coefficient for x[i] * x[j]
# Diagonal Q[i][j] is coefficient for x[i] (linear term)

Q = np.zeros((4, 4))
# Fill Q based on travel costs + constraint penalties
# Variables: x0=A@pos1, x1=B@pos1, x2=A@pos2, x3=B@pos2
# Travel (linear): depot->pos1 and pos2->depot
# Transitions (quadratic): pos1->pos2 (A->B and B->A)
# Constraints (penalty p): each position has one customer, each customer appears once

# Diagonal (linear) terms: travel cost + penalty contributions (-2p per variable)
Q[0][0] = 5 - 2 * p   # depot->A (5)
Q[1][1] = 3 - 2 * p   # depot->B (3)
Q[2][2] = 5 - 2 * p   # A->depot (5)
Q[3][3] = 3 - 2 * p   # B->depot (3)

# Quadratic penalty terms (from (sum-1)^2 expansions): 2p for pairs in same constraint
Q[0][1] = 2 * p  # x1*x2 (position 1 constraint)
Q[1][0] = Q[0][1]

Q[2][3] = 2 * p  # x3*x4 (position 2 constraint)
Q[3][2] = Q[2][3]

Q[0][2] = 2 * p  # x1*x3 (customer A constraint)
Q[2][0] = Q[0][2]

Q[1][3] = 2 * p  # x2*x4 (customer B constraint)
Q[3][1] = Q[1][3]

# Transition costs between positions (travel between customers)
Q[0][3] = 4  # A@pos1 -> B@pos2
Q[3][0] = Q[0][3]

Q[1][2] = 4  # B@pos1 -> A@pos2
Q[2][1] = Q[1][2]

# Q is now filled

print("QUBO Matrix Q:")
print(Q)





# ============================================
# BRUTE FORCE VERIFICATION
# ============================================

def decode_solution(x):
    """
    Convert binary variables to human-readable route.
    x = [xA1, xB1, xA2, xB2] where:
    xA1 = A at position 1, xB1 = B at position 1
    xA2 = A at position 2, xB2 = B at position 2
    """
    # Find who is at position 1
    pos1 = None
    if x[0] == 1:  # xA1
        pos1 = 'A'
    elif x[1] == 1:  # xB1
        pos1 = 'B'
    
    # Find who is at position 2
    pos2 = None
    if x[2] == 1:  # xA2
        pos2 = 'A'
    elif x[3] == 1:  # xB2
        pos2 = 'B'
    
    if pos1 and pos2:
        return f"Depot->{pos1}->{pos2}->Depot"
    return "Invalid/Incomplete"

def check_constraints(x):
    """
    Verify hard constraints:
    1. Exactly one customer at position 1
    2. Exactly one customer at position 2  
    3. Customer A appears exactly once
    4. Customer B appears exactly once
    """
    # Position constraints
    pos1_count = x[0] + x[1]  # A at 1 + B at 1
    pos2_count = x[2] + x[3]  # A at 2 + B at 2
    
    # Customer constraints  
    a_count = x[0] + x[2]  # A at 1 + A at 2
    b_count = x[1] + x[3]  # B at 1 + B at 2
    
    return (pos1_count == 1 and pos2_count == 1 and 
            a_count == 1 and b_count == 1)

def calculate_route_cost(x, dist):
    """Calculate actual travel distance for a valid solution."""
    if not check_constraints(x):
        return float('inf')
    
    # Find the route
    route = []
    if x[0] == 1: route.append(1)  # A first
    elif x[1] == 1: route.append(2)  # B first
    
    if x[2] == 1: route.append(1)  # A second
    elif x[3] == 1: route.append(2)  # B second
    
    if len(route) != 2:
        return float('inf')
    
    # Calculate: depot -> first -> second -> depot
    cost = (dist[0][route[0]] +  # depot to first
            dist[route[0]][route[1]] +  # first to second
            dist[route[1]][0])  # second to depot
    
    return cost

# Enumerate all 16 possible bitstrings
print("\n" + "="*70)
print("BRUTE FORCE ENUMERATION (All 16 possible solutions)")
print("="*70)
print(f"{'Bits':>6} | {'x':>14} | {'Energy':>8} | {'Valid?':>6} | {'Route':>20} | {'True Cost':>9}")
print("-"*70)

best_energy = float('inf')
best_solution = None
valid_solutions = []

for bits in range(16):
    x = np.array([(bits >> i) & 1 for i in range(4)])
    energy = x @ Q @ x  # x^T * Q * x
    valid = check_constraints(x)
    route = decode_solution(x)
    true_cost = calculate_route_cost(x, dist) if valid else float('inf')
    
    valid_str = "YES" if valid else "NO"
    print(f"{bits:>06b} | {str(x):>14} | {energy:>8.1f} | {valid_str:>6} | {route:>20} | {true_cost if valid else 'N/A':>9}")
    
    if valid:
        valid_solutions.append((bits, x, energy, route, true_cost))
        if energy < best_energy:
            best_energy = energy
            best_solution = (bits, x, energy, route, true_cost)

print("-"*70)
print("\n" + "="*70)
print("SUMMARY: VALID SOLUTIONS ONLY")
print("="*70)
for bits, x, energy, route, true_cost in valid_solutions:
    print(f"Route: {route:25} | QUBO Energy: {energy:6.1f} | True Distance: {true_cost}")

print(f"\n{'='*70}")
print(f"OPTIMAL SOLUTION FOUND:")
print(f"  Bitstring: {best_solution[0]:04b}")
print(f"  Variables: {best_solution[1]}")
print(f"  Route: {best_solution[3]}")
print(f"  QUBO Energy: {best_solution[2]:.1f}")
print(f"  True Distance: {best_solution[4]}")
print(f"{'='*70}")

# Verification check
expected_optimal = "Depot->B->A->Depot"  # Cost 3+4+5 = 12
expected_cost = 12
if best_solution[3] == expected_optimal and best_solution[4] == expected_cost:
    print("\n✓ VERIFICATION PASSED: QUBO correctly finds optimal route!")
else:
    print(f"\n✗ VERIFICATION FAILED: Expected {expected_optimal} (cost {expected_cost})")
    print("  Check your QUBO formulation.")


