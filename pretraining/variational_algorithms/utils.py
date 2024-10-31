from typing import List, Tuple, Union

import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit import library as gates

try: # For Qiskit < 1.0:
    from qiskit.quantum_info.synthesis.two_qubit_decompose import TwoQubitWeylDecomposition
except: # For Qiskit >=1.0:
    from qiskit.synthesis import TwoQubitWeylDecomposition

from variational_algorithms.hamiltonians import X, Y, Z, I, PauliOperator, Hamiltonian


class KAKDecomposition:
    """Performs the KAK Decomposition to a given 4x4 unitary matrix. Given a two-qubit quantum gate it is decomposed as
    U = a0xa1 * exp(i(x*XX + y*YY + z*ZZ)) b0xb1, where a0, a1, b0, b1 are single-qubit unitary gates and XX, YY, ZZ
    correspond to the corresponding Pauli matrices acting on both qubits with some parameters x, y, z.
    If a 2x2 matrix is given it just looks for the corresponding U3 rotation angles.

    Methods:
        - add_to_circuit(): adds the decomposed gates into the specified circuit object.

    Attributes:
        - b0, b1: single qubit gates resulting from the decomposition applied before the two-qubit gates
        - a0, a1: single qubit gates resulting from the decomposition applied after the two-qubit gates
        - x, y, z: parameters for the RXX, RYY and RZZ gates
        - g_phase: global phase resulting from the decomposition
        - angles: dictionary of angles for each gate where keys 'b0', 'b1', 'a0', 'a1' (or 'u3' in the case of a single-
        qubit gate) return the angles as (theta, phi, lamda) and keys 'rxx', 'ryy', 'rzz' return the angles x, y, z
        respectively."""

    def __init__(self, unitary: np.ndarray) -> None:
        """Performs the KAK Decomposition on a 4x4 unitary matrix and computes the rotation angles for the single-qubit
        gates. If a 2x2 unitary matrix is passed it only gets the rotation angles for the single-qubit gate.

        Args:
            unitary (np.ndarray): Unitary matrix to be decomposed.
        """
        if unitary.shape == (2, 2):
            self._nqubits = 1
            self.angles = {"u3": self._get_u3_angles(unitary)}
        elif unitary.shape == (4, 4):
            self._nqubits = 2
            self._kak_decomp(unitary)
        else:
            raise ValueError("Input must be a 2x2 or 4x4 matrix.")

    def _kak_decomp(self, unitary: np.ndarray):
        decomp = TwoQubitWeylDecomposition(unitary)
        self.b0, self.b1 = decomp.K2l, decomp.K2r
        self.a0, self.a1 = decomp.K1l, decomp.K1r
        self.x, self.y, self.z = -2 * decomp.a, -2 * decomp.b, -2 * decomp.c
        self.g_phase = np.exp(1j * decomp.global_phase)

        self.angles = self._get_angles_dict()

    def _get_angles_dict(self):
        return {
            **{i: self._get_u3_angles(getattr(self, i)) for i in ("b0", "b1", "a0", "a1")},
            **{f"r{2*i}": (getattr(self, i),) for i in ("x", "y", "z")},
        }

    def _get_u3_angles(self, unitary: np.ndarray):
        unitary = unitary / np.exp(1j * np.angle(unitary[0, 0]))
        theta = np.real(2 * np.arccos(unitary[0, 0]))
        if abs(np.sin(theta / 2)) > 1e-15:
            phi = np.real(-1j * np.log(unitary[1, 0] / np.sin(theta / 2)))
            lam = np.real(-1j * np.log(-unitary[0, 1] / np.sin(theta / 2)))
        else:
            phi = np.real(-1j * np.log(unitary[1, 1] / np.cos(theta / 2)))
            lam = 0
        return (theta, phi, lam)

    def add_to_circuit(self, circuit: QuantumCircuit, qubits: Tuple[int]):
        """Adds the gates resulting from the decomposition into the specified quantum circuit.

        Args:
            circuit (QuantumCircuit): Quantum circuit where the gates will be added
            backend (Backend): Backend used for the circuit simulation
            qubits (Tuple[int]): Qubits of the circuit where the gates will be added

        Returns:
            circuit (QuantumCircuit): Quantum circuit updated with the gates resulting from the decomposition
            angles_list (List): List of the angles used in the parametrized gates added, sorted by the order in which
            the gates are added.
        """
        if len(qubits) != self._nqubits:
            raise ValueError("The number of qubits specified doesn't correspond with the size of the gate.")

        if self._nqubits == 1:
            angles_list = list(self.angles)
            circuit.append(gates.UGate(*self.angles["u3"]), [qubits[0]])
        else:
            angles_list = [ang for gate in ["b0", "b1", "rxx", "ryy", "rzz", "a0", "a1"] for ang in self.angles[gate]]
            circuit.append(gates.UGate(*self.angles["b0"]), [qubits[0]])
            circuit.append(gates.UGate(*self.angles["b1"]), [qubits[1]])
            circuit.append(gates.RXXGate(*self.angles["rxx"]), list(qubits))
            circuit.append(gates.RYYGate(*self.angles["ryy"]), list(qubits))
            circuit.append(gates.RZZGate(*self.angles["rzz"]), list(qubits))
            circuit.append(gates.UGate(*self.angles["a0"]), [qubits[0]])
            circuit.append(gates.UGate(*self.angles["a1"]), [qubits[1]])

        return circuit, angles_list


def hermiticity_check(operator: PauliOperator | Hamiltonian | int | float | complex) -> bool:
    """
    Checks whether an operator is hermitian by looking for complex coefficients. It assumes that it is constructed
    with PauliOperators, which are hermitian themselves.

    Args:
        operator (PauliOperator | Hamiltonian | int | float | complex): The operator to be checked.

    Returns:
        bool: Whether the operator is hermitian (True) or not (False).
    """
    if isinstance(operator, (int, float)):
        return True

    elif isinstance(operator, complex):
        return False

    elif isinstance(operator, (PauliOperator, Hamiltonian)):
        for term in operator.parse():
            if isinstance(term[0], complex):
                return False
        return True
