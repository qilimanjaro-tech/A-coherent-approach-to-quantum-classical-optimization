from abc import abstractmethod
from typing import Union, List

import numpy as np

from qiskit.circuit import QuantumCircuit

import variational_algorithms.hamiltonians as sym
from variational_algorithms.hamiltonians import Hamiltonian


class Results:
    """
    Class representing the results obtained from executing circuits.

    Attributes:
        frequencies (dict[tuple[int], int]): A dictionary mapping measurement outcomes to their frequencies.
        state (list): A list representing the quantum state obtained from the backend.

    Methods:
        get_frequencies: Returns the frequencies of measurement outcomes.
        get_state: Returns the quantum state obtained from the backend.

    """

    def __init__(self, frequencies: dict[tuple[int], int], state: list):
        self.frequencies = frequencies
        self.state = state

    def get_frequencies(self) -> dict[tuple[int], int]:
        """
        Get the frequencies of measurement outcomes

        Returns:
            dict[tuple[int], int]: A dictionary mapping measurement outcomes to their frequencies.

        """

        return self.frequencies

    def get_state(self) -> list:
        """
        Gets the output quantum state of the circuit.

        Returns:
            list: The quantum state in array-like format of length 2**nqubits containing the coefficients in the measurement basis.

        """

        return self.state


class Operator:
    """
    Represents a quantum operator.

    Parameters:
        operator (hamiltonians.Hamiltonian or hamiltonians.PauliOperator, int, float or complex): The quantum operator.
        If the operator is of type int, float or complex it is converted into a Hamiltonian object.
        nqubits (int): The number of qubits on which the operator acts on.

    Raises:
        TypeError: If the operator is not of type Hamiltonian, PauliOperator, int, complex or float.

    Attributes:
        op (hamiltonians.Hamiltonian or hamiltonians.PauliOperator): The quantum operator.
        nqubits (int): The number of qubits.

    Methods:
        get_op(): Returns the quantum operator in the Hamiltonian format.

    """

    def __init__(self, operator: sym.Hamiltonian | sym.PauliOperator | int | float | complex, nqubits: int) -> None:
        self.nqubits = nqubits

        if isinstance(operator, (int, float, complex)):
            self.op = sym.Hamiltonian([operator], sym.Operation.ADD)

        elif isinstance(operator, (sym.Hamiltonian, sym.PauliOperator)):
            self.op = operator

        else:
            raise TypeError(
                f"Incompatible types: Expected Hamiltonian, PauliOperator, int, float or complex object and received {type(operator)}"
            )

    def get_op(self) -> Hamiltonian:
        """
        Gets the quantum operator in the Hamiltonian format.

        Returns:
            hamiltonians.Hamiltonian: The quantum operator in the Hamiltonian format.
        """

        return self.op

    def _diagonalize_pauli_term(
        self, circuit, term: tuple[complex, list[sym.PauliOperator]], inverse: bool = False
    ) -> tuple[complex, list[int], QuantumCircuit]:
        """
        Creates the circuit that rotates the measurement basis to the eigenbasis of the given Pauli term.

        Args:
            term (tuple[complex, List[hamiltonians.PauliOperator]]): The Pauli term to be diagonalized.
            inverse (bool): Whether to return the circuit that applies the inverse operation to the diagonalization.

        Returns:
            tuple[complex, list[int], QuantumCircuit]: The coefficient, target qubits, and diagonalizing.
            QuantumCircuit circuit objects.
        """

        coefficient, pauli_operators = term

        targets = []
        for pauli in pauli_operators:
            target = pauli.qubit_id
            if isinstance(pauli, sym.I):
                pass

            elif isinstance(pauli, sym.Z):
                targets.append(target)

            elif isinstance(pauli, sym.X):
                circuit.h(target)
                # circ.append(H(target))
                targets.append(target)

            elif isinstance(pauli, sym.Y):
                if not inverse:
                    circuit.h(target)
                    circuit.rz((-1 + 2 * int(inverse)) * np.pi / 2, target)
                    # circ.append(H(target))
                    # circ.append(RZ(target, (-1 + 2 * int(inverse)) * np.pi / 2))

                else:
                    circuit.rz((-1 + 2 * int(inverse)) * np.pi / 2, target)
                    circuit.h(target)
                    # circ.append(RZ(target, (-1 + 2 * int(inverse)) * np.pi / 2))
                    # circ.append(H(target))

                targets.append(target)
                coefficient *= -1

        return (coefficient, targets, circuit)

    def _pauli_expectation_circuits(self) -> List[tuple[complex, List[int], QuantumCircuit]]:
        """
        Generates a list of circuits that when appended to a given circuit that allow for the computation of expected
        values of non diagonal operators.

        Returns:
            List[tuple[complex, list[int], QuantumCircuit]]: The list of diagonalizing circuits.
        """

        circ_list = []
        for term in self.op.parse():
            coefficient, targets, circ = self._diagonalize_pauli_term(term)
            circ_list.append((coefficient, targets, circ))
        return circ_list

    def _commuting_evol_circuit(self, circuit, dt: float) -> QuantumCircuit:
        """
        Constructs a quantum circuit assuming an operator with commuting Pauli terms.

        Args:
            dt (float): The time step for the evolution.

        Returns:
            QuantumCircuit: The quantum circuit for exact evolution.
        """
        for term in self.op.parse():
            if (len(term[1]) == 1 and isinstance(term[1][0], sym.I)) or len(term[1]) == 0:
                pass

            elif len(term[1]) == 1 and not isinstance(term[1][0], sym.I):
                coefficient, targets, circuit = self._diagonalize_pauli_term(circuit, term, inverse=True)
                circuit.rz(2 * coefficient * dt, targets[0])
                _, _, circuit = self._diagonalize_pauli_term(circuit, term, inverse=False)

                # evol_circ.append(diag_circ)
                # evol_circ.append(RZ(targets[0], 2 * coefficient * dt))
                # evol_circ.append(self._diagonalize_pauli_term(term, inverse=False)[2])

            elif len(term[1]) > 1:
                coefficient, targets, circuit = self._diagonalize_pauli_term(circuit, term, inverse=True)
                for i in range(1, len(targets)):
                    circuit.cx(targets[i - 1], targets[i])
                circuit.rz(2 * coefficient * dt, targets[-1])
                for i in range(1, len(targets)):
                    circuit.cx(targets[-(i + 1)], targets[-i])
                _, _, circuit = self._diagonalize_pauli_term(circuit, term, inverse=False)

                # evol_circ.append(diag_circ)
                # evol_circ.append([CNOT(targets[i - 1], targets[i]) for i in range(1, len(targets))])
                # evol_circ.append(RZ(targets[-1], 2 * coefficient * dt))
                # evol_circ.append([CNOT(targets[-(i + 1)], targets[-i]) for i in range(1, len(targets))])
                # evol_circ.append(self._diagonalize_pauli_term(term, inverse=False)[2])

        return circuit

    def trotter_circuit(self, circuit, dt: float, order: int = 1, trotter_steps: int = 1) -> QuantumCircuit:
        """
        Generates a circuit for Trotterization of time evolution.
        Args:
            dt (float): The time step of the evolution.
            order (int): The order of the Trotter Approximation. Defaults to 1.
            trotter_steps (int): The number of Trotter steps. Defaults to 1.

        Returns:
            QuantumCircuit: The trotterized evolution circuit.

        Raises:
            ValueError: If the given hamiltonian is non-hermitian. It assumes that it is constructed out of Pauli
            operators which are hermitian themselves.
            NotImplementedError: If the given order is higher than 1. Higher order Trotterization is not implemented yet.
        """

        if any(isinstance(term[0], complex) for term in self.op.parse()):
            raise ValueError('Given "hamiltonian" contains complex coefficients, thus being non hermitian.')

        if order != 1:
            raise NotImplementedError("Trotterization of order higher than 1 is not implemented yet.")

        for _ in range(trotter_steps):
            circuit = self._commuting_evol_circuit(circuit, dt / trotter_steps)

        return circuit

    def exact_evol_circuit(self, dt: float) -> QuantumCircuit:
        """
        Constructs a quantum circuit for exact evolution given by an operator with commuting Pauli terms.
        Args:
            dt (float): The time step for the evolution.
        Returns:
            QuantumCircuit: The quantum circuit for exact evolution.
        """
        op_dict = None

        for term in self.op.parse():
            if isinstance(term[0], complex):
                raise ValueError("Operator is not hermitian")

            if op_dict is None:
                op_dict = {pauli.qubit_id: pauli for pauli in term[1]}
            else:
                if any(
                    [
                        (pauli.qubit_id in op_dict.keys())
                        and not isinstance(pauli, type(op_dict[pauli.qubit_id]))
                        and not isinstance(pauli, sym.I)
                        for pauli in term[1]
                    ]
                ):
                    raise ValueError("Given Hamiltonian contains non commuting terms")
                for pauli in term[1]:
                    if pauli.qubit_id not in op_dict.keys():
                        op_dict[pauli.qubit_id] = pauli

        return self._commuting_evol_circuit(dt)
