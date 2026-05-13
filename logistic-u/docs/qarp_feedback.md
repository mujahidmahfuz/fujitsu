# QARP SDK Feedback Document

**Team:** Quantum Logistics — Tokyo SME Delivery VRP  
**Date:** February 2026  
**Challenge:** Fujitsu Quantum Simulator Challenge 2025-26

---

## Executive Summary

We developed a modular quantum-enhanced Vehicle Routing Problem (VRP) solution
targeting Tokyo's Shibuya Ward using QUBO formulations solved via Grover Adaptive
Search and QAOA on the Fujitsu quantum simulator. Below is our experience-based
feedback on the QARP SDK, organized by usability, features, and recommendations.

---

## 1. What Worked Well

### 1.1 Problem Type Support
The QARP SDK's native support for QUBO problems is well-aligned with combinatorial
optimization use cases like VRP. The Ising → QARP format conversion is
straightforward when provided clear documentation.

### 1.2 Cloud Execution Model
The job submission → polling → result retrieval flow is clean and familiar to
users of cloud quantum platforms (IBM Quantum, Amazon Braket). The JSON-based
problem export format is interoperable and easy to debug.

### 1.3 Simulator Scale
The ability to simulate 36+ qubits sets the QARP platform apart from gate-based
simulators. For VRP, this means we can tackle 8-stop instances directly (8 stops
× 4 positions = 32 qubits with position encoding), which is commercially relevant
for last-mile delivery.

---

## 2. Challenges Encountered

### 2.1 Limited Documentation for Hybrid Workflows
Our solver pipeline chains **QAOA → Grover Adaptive Search** (warm-started).
The QARP SDK documentation primarily covers single-shot optimization, without
clear guidance on:
- How to extract intermediate quantum state information for warm-starting
- Whether custom operator sequences (e.g., Grover oracles on QUBO cost functions)
  are supported or must be decomposed into native gates
- Circuit depth limits and how they relate to VRP problem size

**Recommendation:** Provide examples showing how to implement multi-stage
quantum algorithms (e.g., variational + search hybrids) on QARP.

### 2.2 Encoding Guidance Missing
VRP problems can use multiple QUBO encodings (position-based O(n²), route-based
O(n log n)). The optimal encoding depends on the simulator architecture:
- **Position encoding:** Dense QUBO, more qubits, simpler constraints
- **Route encoding:** Sparser QUBO, fewer qubits, complex constraints

**Recommendation:** Document how QARP's tensor-network simulator handles QUBO
density/sparsity. This would help users select encodings that exploit the
simulator's strengths (e.g., low-entanglement states → high bond dimension
efficiency in MPS).

### 2.3 Penalty Weight Sensitivity
QUBO penalty weights critically affect solution quality. Our analysis shows:
- Too low (0.1-0.25x): QUBO ground state violates constraints
- Too high (5-10x): Energy landscape flattens, search algorithms stall
- Sweet spot: 1.0-2.0x auto-calibrated value

The QARP SDK could provide built-in penalty calibration tools or at least
document how simulator noise/precision affects optimal penalty ranges.

### 2.4 Circuit Cutting Integration
For instances exceeding 36 qubits, we implemented circuit cutting (spectral
QUBO partitioning). However, the QARP SDK does not natively support:
- Fragment submission as sub-problems
- Automatic partition-based job splitting
- Classical recombination of fragment results

**Recommendation:** Add a circuit-cutting API layer that accepts large QUBOs
and automatically partitions, solves fragments, and recombines.

---

## 3. Feature Requests

### 3.1 Real-Time Re-optimization API
For logistics applications, routes must be re-optimized mid-delivery (traffic,
accidents, customer changes). A streaming API that accepts updated QUBOs and
returns updated solutions with warm-started parameters would be transformative.

### 3.2 Benchmark Problem Library
Provide standardized benchmark QUBOs (TSP, VRP, MaxCut, portfolio optimization)
at various scales. This would help users:
- Validate their setups before custom problems
- Compare solver performance across platforms
- Learn QARP best practices from working examples

### 3.3 MPS/Tensor-Network Controls
Expose tensor-network simulation parameters (bond dimension, truncation error,
convergence criteria) to advanced users. This enables:
- Speed vs accuracy trade-offs for time-critical applications
- Convergence diagnostics for debugging
- Fair benchmarking against other tensor-network libraries

### 3.4 Solution Quality Metrics
Return solution quality metadata alongside the bitstring:
- Approximation ratio vs known classical bounds
- Constraint satisfaction status
- Energy convergence history

---

## 4. Integration Architecture

Our final integration layer (`qarp_integration.py`) provides:

```python
class QARPInterface:
    def solve_qubo(self, qubo: QUBOResult) -> dict:
        """Submit QUBO to QARP, falls back to local simulator."""
        ...
    
    def convert_to_qarp_format(self, qubo: QUBOResult) -> dict:
        """QUBO → Ising → QARP-compatible JSON."""
        ...
    
    def export_problem(self, qubo: QUBOResult, filepath: str):
        """Export problem for offline submission."""
        ...
```

This abstraction allows seamless switching between local numpy simulation and
QARP cloud execution.

---

## 5. Summary

| Category | Rating | Notes |
|----------|--------|-------|
| Problem format support | ⭐⭐⭐⭐ | QUBO/Ising well-supported |
| Documentation | ⭐⭐ | Needs hybrid workflow examples |
| Scalability | ⭐⭐⭐⭐⭐ | 36+ qubits is industry-leading |
| Advanced features | ⭐⭐ | Missing circuit cutting, warm-start |
| SDK ergonomics | ⭐⭐⭐ | Clean API, but needs more examples |

**Overall: 3.5/5** — Promising platform with strong scalability. Key gaps
in documentation and advanced algorithm support that, if addressed, would
make QARP the go-to platform for quantum optimization.
