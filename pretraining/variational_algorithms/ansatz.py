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


class HardwareEfficientAnsatz(Ansatz):
    """
    Class that allows to implement parameterized quantum circuits (PQC) that are efficient for NISQ hardware. 

    Args:
    - n_qubits (int): indicates the number of qubits of the ansatz.
    - layers (int, optional): indicates the number of efficient hardware layers the ansatz possesses.
    - connectivity (literal, optional): indicates the type of hardware connectivity, how the two-qubit gates are connected.
    - structure (literal, optional): indicates the way in which the two-qubit and one-qubit gates are inserted in the anstaz.
    - one_gate (literal, optional): indicates the type of gate used as one gate.
    - two_gate (literal, optional): indicates the type of gate used as a two gate.
    """
    def __init__(
        self,
        n_qubits: int,
        layers: Optional[int] = 1,
        connectivity: Union[Literal["Circular"], Literal["Linear"], Literal["Full"], List[Tuple[int, int]]] = "Linear",
        one_gate: Union[Literal["U1"], Literal["U2"], Literal["U3"]] = "U1",
        two_gate: Union[Literal["CZ"], Literal["CNOT"]] = "CZ",
        ini_circuit: QuantumCircuit = None,
        init_hadam: bool = False
    ):
        super(HardwareEfficientAnsatz, self).__init__(n_qubits, layers)

        # Define chip topology
        if isinstance(connectivity, list):
            self.connectivity = connectivity
        else:
            if connectivity == "Full":
                self.connectivity = [(i, j) for i in range(self.n_qubits) for j in range(i + 1, self.n_qubits)]
            else:
                self.connectivity = [(i, i + 1) for i in range(self.n_qubits - 1)] + (
                    [(self.n_qubits - 1, 0)] if connectivity == "Circular" else []
                )

        self.params_per_unitary = self.nparams_one_gates_dict[one_gate]
        self.one_gate = one_gate
        self.two_gate = two_gate
        self.layers = layers
        self.total_params = self.n_qubits * layers * self.params_per_unitary

        self.ini_circuit = ini_circuit
        self.init_hadam = init_hadam

    @property
    def number_of_parameters(self) -> int:
        return self.total_params

    def construct_circuit(self, params: List[float]):
        parameters = list(params.copy())
        # Assert number of parameters is correct:
        assert len(parameters) >= self.total_params
        if not self.ini_circuit:
            self.circuit = QuantumCircuit(self.n_qubits)
        else:
            self.circuit = deepcopy(self.ini_circuit)

        if self.init_hadam:
            self.circuit.h(range(self.n_qubits))

        # Add as many layers as desired
        construct_layer = self.construct_layer_grouped
        for _ in range(self.layers):
            parameters = construct_layer(parameters)

        self.circuit.measure(self.circuit.qregs[0], self.circuit.cregs[0])
        return self.circuit

    def construct_layer_grouped(self, params: List[float]) -> List[float]:
        for i in range(self.n_qubits):
            unit_params = tuple(params.pop() for _ in range(self.params_per_unitary))
            self.add_U(i, unit_params)

        for i, j in self.connectivity:
            self.add_U_2((i,j))

        return params


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

class CDQAOAAnsatz(Ansatz):
    """
    Class representing the ansatz for the CDQAOA algorithm. The time evolution circuits used to apply the
    Hamiltonian exponentials are constructed using the first order Trotter-Suzuki decomposition with a single step. For
    Hamiltonians whose terms commute, the they perform the same as the exact time evolution. For non-commuting terms, the

    Args:
        n_qubits (int): The number of qubits in the ansatz circuit.
        layers (int): Number of layers of mixing and cost hamiltonian exponentials in the ansatz.
        costfunction (CostFunction): The cost function that contains the cost hamiltonian.
        order (int): order for the nested commutator for calculating the CD terms
        k_body (int): maximum number of interactions for the CD terms
        custom_mixer_hamiltonian (Hamiltonian | PauliOperator | int | float | complex, optional): An optional
        custom_mixer hamiltonian to be used in the ansatz. By default the regualr sum of X gates is used. Defaults to
        None.
    """

    def __init__(
        self,
        n_qubits: int,
        layers: int,
        costfunction: CostFunction,
        order: int = 1,
        k_body: int = 1,
        custom_mixer_hamiltonian: Hamiltonian | PauliOperator | int | float | complex = None,
        ini_circuit: QuantumCircuit = None,
    ):
        super(CDQAOAAnsatz, self).__init__(n_qubits, layers)

        self.cost_ham = costfunction.total_hamiltonian
        self._setup_custom_mixer(custom_mixer_hamiltonian)

        self._generate_CD_hamiltonian(order,k_body)

        self._params_per_layer = 2+len(self.CD_ham.elements)
        self._total_params = layers*self._params_per_layer

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

    def _generate_CD_hamiltonian(self,order,k_body) -> None:
        # Calculate the CD terms for the nested commutator

        Ha = self.mixer+self.cost_ham
        dHa = self.mixer-self.cost_ham

        # Calculate nested commutator at 1st order
        prev = Ha*dHa-dHa*Ha
        self.CD_ham = (0+1j)*prev

        # Calculate higher order nested commutators
        for i in range(order-1):
            prev = Ha*Ha*prev-2*Ha*prev*Ha+prev*Ha*Ha
            self.CD_ham += (0+1j)*prev
        
        # Clean higher k-body terms
        for term,strength in self.CD_ham.elements.items():
            if len(term)>k_body:
                self.CD_ham-= Hamiltonian({term:strength})
        
        # Clean the deleted terms
        self.CD_ham.filter()

    @property
    def number_of_parameters(self) -> int:
        return self._total_params

    def _add_layer(
        self,
        param1: float,
        param2: float,
        params: float | list[float],
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

        c = 0

        for term,strength in self.CD_ham.elements.items():
            self.circuit = Operator(Hamiltonian({term:strength.real}),self.n_qubits).trotter_circuit(self.circuit, params[c])
            c+=1

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
            self._add_layer(params[self._params_per_layer*i], params[self._params_per_layer*i+1], params[self._params_per_layer*i+2:self._params_per_layer*(i+1)])

        self.circuit.measure(self.circuit.qregs[0], self.circuit.cregs[0])

        return self.circuit
