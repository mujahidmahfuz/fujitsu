"""
Lesson 1: The first Qubit
Fujitsu Challenge - Champion's Path
Date: feb-2, 2026
Student: MD MUJAHIDUL ISLAM (MUJAHID)

"""


from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


#Step 1: Create a circuit with 1 qubit, 1 classical bit
qc = QuantumCircuit(1, 1)

#Step 2: Put qubit in superposition (|0> + |1>/sqrt(2))
qc.h(0)  # Hadamard gate


#step 3: Measure
qc.measure(0,0)


# Step 4: Simulate
simulator = AerSimulator(method='statevector')
compiled = transpile(qc, simulator)
job = simulator.run(compiled, shots=1024)
result = job.result()
counts = result.get_counts()


#Step 5: Display
print("Measurement Results:", counts)
print("Expected: ~50% '0', ~50% '1'")
print("Superposition Verified!" if abs(counts.get('0', 0) - counts.get('1', 0)) < 100 else "Error in Superposition")


# Save visualization
fig = plot_histogram(counts)
plt.savefig('lesson1_superposition.png')
print("Visualization saved to: lesson1_superposition.png")