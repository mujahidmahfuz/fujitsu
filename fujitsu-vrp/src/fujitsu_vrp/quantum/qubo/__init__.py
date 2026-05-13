"""QUBO encoding and constraint modules."""

from .encoder import EncodingConfig, QUBOProblem, VRP_QUBOEncoder
from .constraints import (
    ConstraintPenalties,
    assignment_constraint_hamiltonian,
    slot_uniqueness_constraint,
    capacity_constraint_hamiltonian,
    time_window_constraint_hamiltonian,
    objective_hamiltonian,
    build_full_hamiltonian,
)
from .penalty_tuning import PenaltyTuner, PenaltyTuningConfig, estimate_penalty_range
from .decomposition import ProblemDecomposer, Decomposition, Subproblem

__all__ = [
    "EncodingConfig",
    "QUBOProblem",
    "VRP_QUBOEncoder",
    "ConstraintPenalties",
    "assignment_constraint_hamiltonian",
    "slot_uniqueness_constraint",
    "capacity_constraint_hamiltonian",
    "time_window_constraint_hamiltonian",
    "objective_hamiltonian",
    "build_full_hamiltonian",
    "PenaltyTuner",
    "PenaltyTuningConfig",
    "estimate_penalty_range",
    "ProblemDecomposer",
    "Decomposition",
    "Subproblem",
]