from typing import Dict, List, Union

from qiskit.circuit import QuantumCircuit
from qiskit import transpile

from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime import SamplerV2 as SamplerQiskit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from variational_algorithms.backend import Results


class Sampler:
    def __init__(self, backend=None, 
                 session= None, 
                 optimization_level=1, 
                 routing_method ='basic', 
                 layout_method = 'trivial',
                 error_mitigation =  False,
                 *args, 
                 **kwargs):
        
        self.backend = backend
        self.session = session
        self.optimization_level = optimization_level
        self.routing_method = routing_method 
        self.layout_method = layout_method 
        self.error_mitigation = error_mitigation
        self.circuit = None
        self.parameters = []
        self.quantum_state = None
        self.probability_dict = None
        self.required_qubits = None

    @staticmethod
    def number_of_parameters():
        "A static method that returns the number of parameters needed by the Sampler."
        return 0

    def set_circuit(self, circuit: QuantumCircuit):
        """Updates the circuit stored in the sampler.
        Args:
            - circuit (QuantumCircuit): the new circuit to be added to the sampler.
        """
        if self.required_qubits is not None and circuit.num_qubits != self.required_qubits:
            raise ValueError(f"Sampler requires a circuit with {self.required_qubits} qubits.")
        self.circuit = circuit

    def set_parameters(self, parameters: List[float]):
        """Updates the sampler parameters.
        Args:
            - parameters (List[float]): the list of new parameters.
        """
        self.parameters = parameters

    def sample(self, n_shots: int = 1000):
        """Samples the circuit and gets the information about the quantum state and the probabilities of the samples.
        Args:
            - n_shots (int): the number of shots of the quantum circuit sampling.
        """
        if self.backend is None and self.session is None:
            
            backend = AerSimulator(method='statevector')
            qiskit_result = backend.run(self.circuit, shots=n_shots)
            result = Results(frequencies=qiskit_result.result().get_counts(), state=None)

        elif self.backend is not None and self.session is None:
            
            backend = self.backend
            self.circuit = transpile(self.circuit, backend)
            qiskit_result = backend.run(self.circuit, shots=n_shots)
            result = Results(frequencies=qiskit_result.result().get_counts(), state=None)

        elif self.session is not None:
            
            if self.backend is None:
                
                 raise ValueError('It is necessary to specify the backend')
            
            pm = generate_preset_pass_manager(optimization_level=self.optimization_level,
                                              routing_method = self.routing_method,
                                              layout_method = self.layout_method,  
                                              backend=self.backend)
            circuit_transpile = pm.run(self.circuit)

            sampler_qiskit = SamplerQiskit(mode=self.session)
            
            if self.error_mitigation:
                options = sampler_qiskit.options
                options.dynamical_decoupling.enable = True
                options.twirling.enable_gates = True
            
            qiskit_result = sampler_qiskit.run([circuit_transpile], shots=n_shots)
            result = Results(frequencies=getattr(qiskit_result.result()[0].data, self.circuit.cregs[0].name).get_counts(), state=None)

        frequencies = result.get_frequencies()
        self.probability_dict = {i: j / n_shots for i, j in frequencies.items()}

        self.quantum_state = result.get_state()

        return result

    def get_quantum_state(self) -> List[float]:
        """Gets the quantum state of the parsed circuit.
        Returns:
            - the quantum state.
        """
        if self.quantum_state is None:
            raise NotImplementedError()
        return self.quantum_state

    def get_probabilities(self) -> Dict[str, float]:
        """Gets a list of probabilities associated with each state.
        Returns:
            - the probability list in the following format:
                {(binary state, probability), ...}
        """
        if self.probability_dict is None:
            raise NotImplementedError()
        return self.probability_dict
