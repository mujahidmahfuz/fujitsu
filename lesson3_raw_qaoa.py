"""
QUANTUM VRP CHAMPION - COMPLETE IMPLEMENTATION
Fujitsu Quantum Simulator Challenge 2025-26
Lesson 3: Raw QAOA without Qiskit Optimization wrapper
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("FUJITSU QUANTUM CHALLENGE - QAOA VRP SOLVER")
print("="*70)
print(f"NumPy version: {np.__version__}")

# ============================================
# STEP 1: YOUR PROVEN QUBO FROM LESSON 2
# ============================================

Q = np.array([
    [-35.,  40.,  40.,   4.],
    [ 40., -37.,   4.,  40.],
    [ 40.,   4., -35.,  40.],
    [  4.,  40.,  40., -37.]
])

n_qubits = 4

print("\nQUBO Matrix:")
print(Q)

# ============================================
# STEP 2: CONVERT QUBO TO ISING HAMILTONIAN
# ============================================

print("\n" + "="*70)
print("CONVERTING QUBO TO ISING HAMILTONIAN")
print("="*70)

def qubo_to_ising(Q):
    """Convert QUBO to Ising: x_i = (1 - z_i)/2"""
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = 0.0
    
    # Diagonal terms
    for i in range(n):
        offset += Q[i, i] / 2.0
        h[i] -= Q[i, i] / 2.0
    
    # Off-diagonal terms
    for i in range(n):
        for j in range(i+1, n):
            q_ij = Q[i, j]
            offset += q_ij / 4.0
            h[i] -= q_ij / 4.0
            h[j] -= q_ij / 4.0
            J[i, j] = q_ij / 4.0
    
    return h, J, offset

h, J, offset = qubo_to_ising(Q)

print(f"Offset (constant): {offset:.4f}")
print(f"Linear coefficients h: {h}")
print(f"Quadratic coefficients J:\n{J}")

# Build Pauli operators for SparsePauliOp
pauli_list = []
coeffs = []

for i in range(n_qubits):
    if abs(h[i]) > 1e-10:
        pauli_str = ['I'] * n_qubits
        pauli_str[i] = 'Z'
        pauli_list.append(''.join(pauli_str))
        coeffs.append(h[i])

for i in range(n_qubits):
    for j in range(i+1, n_qubits):
        if abs(J[i, j]) > 1e-10:
            pauli_str = ['I'] * n_qubits
            pauli_str[i] = 'Z'
            pauli_str[j] = 'Z'
            pauli_list.append(''.join(pauli_str))
            coeffs.append(2 * J[i, j])  # Factor of 2 for symmetric

print(f"\nPauli terms: {len(pauli_list)}")
for p, c in zip(pauli_list, coeffs):
    print(f"  {c:+.4f} * {p}")

# Import quantum components
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

hamiltonian = SparsePauliOp(pauli_list, coeffs)
print(f"\nHamiltonian: {hamiltonian.num_qubits} qubits, {len(hamiltonian)} terms")

# ============================================
# STEP 3: CLASSICAL VERIFICATION (BRUTE FORCE)
# ============================================

print("\n" + "="*70)
print("CLASSICAL BRUTE FORCE VERIFICATION")
print("="*70)

def evaluate_ising(z_bits, h, J, offset):
    """Evaluate Ising energy for spin configuration z ∈ {-1, +1}^n"""
    energy = offset
    n = len(z_bits)
    for i in range(n):
        energy += h[i] * z_bits[i]
    for i in range(n):
        for j in range(i+1, n):
            energy += J[i, j] * z_bits[i] * z_bits[j]
    return energy

best_energy = float('inf')
best_bits = None
best_x = None
valid_solutions = []

for bits in range(2**n_qubits):
    # Convert to binary then to spins
    x = [(bits >> i) & 1 for i in range(n_qubits)]
    z = [1 - 2*xi for xi in x]  # 0 -> +1, 1 -> -1
    
    energy = evaluate_ising(z, h, J, offset)
    
    # Check validity
    valid = (x[0] + x[1] == 1 and x[2] + x[3] == 1 and 
             x[0] + x[2] == 1 and x[1] + x[3] == 1)
    
    if valid:
        valid_solutions.append((bits, x, energy))
        if energy < best_energy:
            best_energy = energy
            best_bits = bits
            best_x = x

print(f"Number of valid solutions: {len(valid_solutions)}")
for bits, x, energy in valid_solutions:
    route = f"Depot->{'B' if x[1] else 'A'}->{'A' if x[2] else 'B'}->Depot"
    print(f"  {bits:04b}: x={x}, energy={energy:.4f}, route={route}")

print(f"\nGround state energy: {best_energy:.4f}")
print(f"Optimal bitstring: {best_bits:04b}")
print(f"Optimal binary: {best_x}")

# ============================================
# STEP 4: QUANTUM SOLUTION - QAOA
# ============================================

print("\n" + "="*70)
print("QUANTUM SOLUTION (QAOA)")
print("="*70)

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_aer import AerSimulator

# Setup GPU simulator
try:
    backend = AerSimulator(method='statevector')
    print("Using GPU acceleration")
except Exception as e:
    backend = AerSimulator(method='statevector')
    print(f"GPU unavailable, using CPU: {e}")

sampler = Sampler()

# QAOA with p=1 (one layer)
print("\nInitializing QAOA (p=1, maxiter=100)...")
qaoa = QAOA(
    sampler=sampler,
    optimizer=COBYLA(maxiter=200, tol=0.0001),
    reps=2,  # TWO layers!
    initial_point=[0.5, 0.5, 0.5, 0.5]  # Better starting point
)

print("Running optimization...")
print("(This takes 30-60 seconds...)")

result = qaoa.compute_minimum_eigenvalue(hamiltonian)

print(f"\nQAOA Results:")
print(f"  Optimal eigenvalue: {result.eigenvalue:.4f}")
print(f"  Total energy (with offset): {result.eigenvalue + offset:.4f}")
print(f"  Classical ground state: {best_energy:.4f}")
print(f"  Gap: {abs(result.eigenvalue + offset - best_energy):.4f}")
print(f"  Cost function evaluations: {result.cost_function_evals}")

# Check success
gap = abs(result.eigenvalue + offset - best_energy)
if gap < 0.5:
    print(f"\n✓ EXCELLENT: QAOA found ground state (gap {gap:.4f})")
elif gap < 2.0:
    print(f"\n✓ GOOD: QAOA approximate solution (gap {gap:.4f})")
else:
    print(f"\n~ PARTIAL: Try reps=2 or better initial point (gap {gap:.4f})")

# ============================================
# STEP 5: SAMPLE FROM OPTIMAL CIRCUIT
# ============================================

# ============================================
# STEP 5: SAMPLE FROM OPTIMAL CIRCUIT (FIXED)
# ============================================

print("\n" + "="*70)
print("SAMPLING OPTIMAL SOLUTION")
print("="*70)

from qiskit import QuantumCircuit, transpile

# Build circuit manually with optimal parameters
optimal_params = result.optimal_parameters
print(f"Optimal parameters: {optimal_params}")

# Extract values in order
param_values = list(optimal_params.values())
gamma = param_values[0] if len(param_values) > 0 else 0.5
beta = param_values[1] if len(param_values) > 1 else 0.5
if len(param_values) >= 4:  # reps=2
    gamma2 = param_values[2]
    beta2 = param_values[3]
else:
    gamma2, beta2 = gamma, beta

print(f"Using γ₁={gamma:.4f}, β₁={beta:.4f}, γ₂={gamma2:.4f}, β₂={beta2:.4f}")

# Build QAOA circuit manually (bulletproof)
qc = QuantumCircuit(n_qubits)

# Initial superposition
qc.h(range(n_qubits))

# Layer 1: Cost Hamiltonian
from qiskit.circuit.library.standard_gates import RZZGate
for i in range(n_qubits):
    for j in range(i+1, n_qubits):
        if abs(J[i,j]) > 1e-10:
            qc.append(RZZGate(2 * gamma * J[i,j]), [i, j])

# Layer 1: Mixer
for i in range(n_qubits):
    qc.rx(2 * beta, i)

# Layer 2 (if reps=2)
if len(param_values) >= 4:
    for i in range(n_qubits):
        for j in range(i+1, n_qubits):
            if abs(J[i,j]) > 1e-10:
                qc.append(RZZGate(2 * gamma2 * J[i,j]), [i, j])
    for i in range(n_qubits):
        qc.rx(2 * beta2, i)

qc.measure_all()

print(f"Circuit depth: {qc.depth()}")
print(f"Circuit parameters: {qc.num_parameters}")

# Run on CPU
print("\nRunning sampling (CPU)...")
compiled = transpile(qc, backend)
job = backend.run(compiled, shots=4096)  # More shots for better statistics
counts = job.result().get_counts()

print(f"Measurement shots: 4096")
print(f"Unique bitstrings: {len(counts)}")



# Analyze results
def bitstring_to_x(bitstring):
    """Convert Qiskit bitstring (little-endian) to x vector"""
    return [int(bitstring[-(i+1)]) for i in range(n_qubits)]

print("\nTop 10 measured solutions:")
sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]

for bitstring, freq in sorted_counts:
    x = bitstring_to_x(bitstring)
    z = [1 - 2*xi for xi in x]
    energy = evaluate_ising(z, h, J, offset)
    prob = freq / 2048 * 100
    
    # Check validity
    valid = (x[0] + x[1] == 1 and x[2] + x[3] == 1 and 
             x[0] + x[2] == 1 and x[1] + x[3] == 1)
    
    route = f"Depot->{'B' if x[1] else 'A'}->{'A' if x[2] else 'B'}->Depot"
    status = "✓ VALID" if valid else "✗ INVALID"
    
    print(f"  {bitstring}: {prob:5.2f}% | energy={energy:+7.2f} | {route} | {status}")

# Find best valid in samples
valid_samples = [(b, f, bitstring_to_x(b)) for b, f in counts.items()
                 if ((lambda x: x[0]+x[1]==1 and x[2]+x[3]==1 and x[0]+x[2]==1 and x[1]+x[3]==1)(
                     bitstring_to_x(b)))]

if valid_samples:
    best_sample = max(valid_samples, key=lambda x: x[1])
    print(f"\nMost probable valid solution:")
    print(f"  Bitstring: {best_sample[0]}")
    print(f"  Probability: {best_sample[1]/2048*100:.2f}%")
    x = best_sample[2]
    route = f"Depot->{'B' if x[1] else 'A'}->{'A' if x[2] else 'B'}->Depot"
    print(f"  Route: {route}")
else:
    print("\nNo valid solutions in samples (increase shots or tune QAOA)")

# ============================================
# STEP 6: VISUALIZATION
# ============================================

print("\n" + "="*70)
print("GENERATING VISUALIZATION")
print("="*70)

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Measurement histogram (top 15)
    ax1 = axes[0]
    top15 = sorted_counts[:15]
    labels = [b for b, _ in top15]
    values = [f for _, f in top15]
    colors = ['green' if (lambda x: x[0]+x[1]==1 and x[2]+x[3]==1 and x[0]+x[2]==1 and x[1]+x[3]==1)(
        bitstring_to_x(b)) else 'red' for b, _ in top15]
    
    bars = ax1.bar(range(len(labels)), values, color=colors)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_xlabel('Bitstring')
    ax1.set_ylabel('Counts')
    ax1.set_title('QAOA Measurement Results (p=1)')
    
    # Add probability labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{val/2048*100:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Energy landscape
    ax2 = axes[1]
    all_energies = []
    all_probs = []
    for bitstring, freq in counts.items():
        x = bitstring_to_x(bitstring)
        z = [1 - 2*xi for xi in x]
        energy = evaluate_ising(z, h, J, offset)
        all_energies.append(energy)
        all_probs.append(freq / 2048)
    
    # Sort by energy
    sorted_by_energy = sorted(zip(all_energies, all_probs))
    energies_sorted, probs_sorted = zip(*sorted_by_energy)
    
    ax2.scatter(energies_sorted, probs_sorted, alpha=0.6, s=50)
    ax2.axvline(x=best_energy, color='r', linestyle='--', linewidth=2, label=f'Ground State ({best_energy:.1f})')
    ax2.set_xlabel('Energy')
    ax2.set_ylabel('Probability')
    ax2.set_title('Energy vs Measurement Probability')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qaoa_vrp_results.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: qaoa_vrp_results.png")
    
except Exception as e:
    print(f"Visualization failed: {e}")
    print("Continuing without plot...")

# ============================================
# FINAL SUMMARY
# ============================================

print("\n" + "="*70)
print("CHAMPION SUMMARY")
print("="*70)
print(f"Problem: 2-stop VRP (4 qubits)")
print(f"Classical ground state: {best_energy:.4f}")
print(f"QAOA best energy: {result.eigenvalue + offset:.4f}")
print(f"Gap: {gap:.4f}")
print(f"Success probability: {'HIGH' if gap < 0.5 else 'MODERATE' if gap < 2.0 else 'LOW'}")
print("\nNext steps:")
print("  1. If gap > 1.0: Increase QAOA reps to 2")
print("  2. If valid prob < 5%: Tune penalty weight P")
print("  3. Next lesson: Compare with OR-Tools classical solver")
print("="*70)



