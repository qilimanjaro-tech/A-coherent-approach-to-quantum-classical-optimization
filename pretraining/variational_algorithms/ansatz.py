from abc import abstractmethod
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union, Literal

import numpy as np

# from variational_algorithms.gates import CNOT, CZ, U1, U2, U3, M, H
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

from variational_algorithms.backend import Operator
from variational_algorithms.hamiltonians import PauliOperator, Hamiltonian, X
from variational_algorithms.cost_function import CostFunction
from variational_algorithms.utils import hermiticity_check


class Ansatz:
    """
    set_initial_state(init_state: List[float]) -> None:
        Sets the initial state of the qubits.
    """

    def __init__(self, n_qubits: int, layers: int = 1):
        self.n_qubits = n_qubits
        self.layers = layers
        self.total_params = 0
        self.circuit = QuantumCircuit(self.n_qubits)

        self.nparams_one_gates_dict = {"U1": 1, "U2": 2, "U3": 3}
        self.one_gate = None
        self.two_gate = None

    @property
    @abstractmethod
    def number_of_parameters(self) -> int:
        "Returns the number of total parameters required by the ansatz if any."

    def add_U(self, qubit, params):
        """ Adds a single-qubit gate to the circuit"""
        if self.one_gate == 'U1':
            self.circuit.p(params[0], qubit)
        elif self.one_gate == 'U2':
            self.circuit.u(np.pi/2, params[0], params[1], qubit)
        elif self.one_gate == 'U3':
            self.circuit.u(params[0], params[1], params[2], qubit)
        else:
            raise NotImplementedError()

    def add_U_2(self, qubits):
        """ Adds a two-qubit gate to the circuit"""
        if self.two_gate == 'CZ':
            self.circuit.cz(qubits[0], qubits[1])
        elif self.two_gate == 'CNOT':
            self.circuit.cx(qubits[0], qubits[1])
        else:
            raise NotImplementedError()


class QAOAAnsatz(Ansatz):
    """
    Class representing the ansatz for the QAOA algorithm. The time evolution circuits used to apply the
    Hamiltonian exponentials are constructed using the first order Trotter-Suzuki decomposition with a single step. For
    Hamiltonians whose terms commute, the they perform the same as the exact time evolution. For non-commuting terms, the

    Args:
        n_qubits (int): The number of qubits in the ansatz circuit.
        layers (int): Number of layers of mixing and cost hamiltonian exponentials in the ansatz.
        costfunction (CostFunction): The cost function that contains the cost hamiltonian.
        custom_mixer_hamiltonian (Hamiltonian | PauliOperator | int | float | complex, optional): An optional
        custom_mixer hamiltonian to be used in the ansatz. By default the regualr sum of X gates is used. Defaults to
        None.

    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        costfunction: CostFunction,
        custom_mixer_hamiltonian: Hamiltonian | PauliOperator | int | float | complex = None,
        ini_circuit: QuantumCircuit = None,
    ):
        super(QAOAAnsatz, self).__init__(n_qubits, layers)

        self._total_params = 2 * layers
        self.cost_ham = costfunction.total_hamiltonian
        self._setup_custom_mixer(custom_mixer_hamiltonian)

        self.ini_circuit = ini_circuit

        if self.ini_circuit:
            self.prep_circuit = deepcopy(ini_circuit)
            # self.prep_circuit.gates = [gate for gate in self.prep_circuit.gates if not isinstance(gate, M)]

        # self.prep_params = getattr(ini_circuit, "parameters", None)

    def _setup_custom_mixer(self, custom_mixer_hamiltonian: Hamiltonian | PauliOperator) -> None:
        if custom_mixer_hamiltonian:
            if not isinstance(custom_mixer_hamiltonian, (Hamiltonian, PauliOperator)):
                raise ValueError("Custom mixer must be a Hamiltonian or a PauliOperator")
            if not hermiticity_check(custom_mixer_hamiltonian):
                raise ValueError("Custom mixer operator must be hermitian to be a valid hamiltonian.")

            self.mixer = custom_mixer_hamiltonian
        else:
            self.mixer = sum([X(i) for i in range(self.n_qubits)])

    @property
    def number_of_parameters(self) -> int:
        return self._total_params

    def _add_layer(
        self,
        param1: float,
        param2: float,
    ) -> None:
        """
        Add a layer of mixing and cost hamiltonian exponential to the ansatz circuit. The time evolution circuits used
        to apply the Hamiltonian exponentials are constructed using the first order Trotter-Suzuki decomposition, with
        each term being built following the staricase method described in https://arxiv.org/pdf/2305.04807.pdf). In the
        case where all the terms in the hamiltonians commute with each other, the circuits perform exact time evolution.
        used.

        Args:
            param_1 (float): Parameter that gets into the cost hamiltonian exponential operator.
            param_2 (float): Parameter that gets into the mixer hamiltonian exponential operator.
            mixer (Union[PauliOperator, Hamiltonian]): The mixer hamiltonian.
            cost_ham (Union[PauliOperator, Hamiltonian]): The cost hamiltonian.
            mixer_comm (bool): Whether all the terms in the mixer hamiltonian commute with eachother.
            cost_comm (bool): Whether all the terms in the cost hamiltonian commute with eachother.

        Returns:
            QuantumCircuit: Quantum circuit representing a layer of the ansatz, containing a circuit with the
            exponential of  both hamiltonians times their respective parameter.
        """
        self.circuit = Operator(self.cost_ham, self.n_qubits).trotter_circuit(self.circuit, param1)
        self.circuit = Operator(self.mixer, self.n_qubits).trotter_circuit(self.circuit, param2)

    def construct_circuit(
        self,
        params: list[float]
    ):
        """
        Construct a circuit for a QAOA algorithm.

        Args:
            params (ist[float]): List of parameters used in the ansatz circuit.
            ini_state (list[int | float | complex], optional): The initial state to be set into the circuits. Regardless
            of this initial state setting, an initial layer of Hadamard gates is applied to obtain the equal
            superposition in the case of init_state = None. Defaults to None.

        Raises:
            ValueError: If the number of parameters does not match double the number of mixers.

        Returns:
            QuantumCircuit: The ansatz circuit built with the given parameters and hamiltonians.

        """
        if not self.ini_circuit:
            self.circuit = QuantumCircuit(self.n_qubits, self.n_qubits)

            for i in range(self.n_qubits):
                self.circuit.h(i)

        else:
            self.circuit = deepcopy(self.prep_circuit)

        for i in range(self.layers):
            self._add_layer(params[2 * i], params[2 * i + 1])

        self.circuit.measure(self.circuit.qregs[0], self.circuit.cregs[0])

        return self.circuit