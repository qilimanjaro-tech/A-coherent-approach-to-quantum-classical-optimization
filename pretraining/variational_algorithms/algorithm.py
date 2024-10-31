from typing import Dict, List, Optional, Sequence, Union
from copy import deepcopy, copy

from variational_algorithms.ansatz import Ansatz
from variational_algorithms.cost_function import CostFunction
from variational_algorithms.optimizer import Optimizer
from variational_algorithms.sampler import Sampler
from variational_algorithms.use_cases.instance import Instance
from variational_algorithms.logger import Logger

from random import random
from math import pi

from time import time


class VQA:
    def __init__(
        self,
        ansatz: Ansatz,
        cost_function: CostFunction,
        instance: Instance,
        optimizer: Optimizer,
        sampler: Sampler,
        n_shots: int = 1000,
        logger: Logger = None,
    ):
        self.ansatz = ansatz
        self.cost_function = cost_function
        self.instance = instance
        self.optimizer = optimizer
        self.sampler = sampler
        self.n_shots = n_shots
        self.logger = logger
        self._init_time = None
        self.energies_list = []

    @property
    def num_parameters(self):
        return (
            self.ansatz.number_of_parameters
            + self.cost_function.number_of_parameters()
            + self.sampler.number_of_parameters()
        )

    def obtain_probabilities(self, params: Sequence[float]) -> dict:
        """Obtain the probabilities of the circuit.

        Args:
            - params (List[float]): the list of parameters in the current optimization step
        """
        circuit = self.ansatz.construct_circuit(params=params[: self.ansatz.number_of_parameters])
        # backend_circuit = self.backend.build_circuit(self.ansatz.n_qubits)
        # backend_circuit.assemble(circuit)
        num_used_parameters = self.ansatz.number_of_parameters

        self.sampler.set_circuit(circuit=circuit)
        self.sampler.set_parameters(
            params[num_used_parameters : num_used_parameters + self.sampler.number_of_parameters()]
        )
        num_used_parameters += self.sampler.number_of_parameters()

        self.sampler.sample(n_shots=self.n_shots)
        return self.sampler.get_probabilities()

    def evaluate_state(self, params: Sequence[float]) -> float:
        """Evaluates a step of the optimization process.

        Args:
            - params (List[float]): the list of parameters in the current optimization step
        """
        num_used_parameters = self.ansatz.number_of_parameters
        num_used_parameters += self.sampler.number_of_parameters()
        self.cost_function.set_parameters(
            parameters=params[num_used_parameters : num_used_parameters + self.cost_function.number_of_parameters()]
        )

        time0 = time()
        samples = self.obtain_probabilities(params)

        self.cost_function.set_samples(samples)

        num_used_parameters += self.cost_function.number_of_parameters()


        sampling_time = time() - time0

        self.cost_function.set_samples(samples)

        results = self.optimizer.parse_cost(self.cost_function)

        if self.logger is not None:
            aux_params = copy(params)
            if not isinstance(aux_params, list):
                aux_params = aux_params.tolist()
            total_time = time() - (self._init_time if self._init_time is not None else time0)
            self.logger.write({"energy": results, "params": aux_params, "samples": samples,
                            'sampling_time': sampling_time, 'tot_accum_time': total_time})

        self.energies_list.append(results)

        return results

    def solve(self, init_params: List[float]=None, **kwargs):
        """Optimizes a given ansatz according to the specified cost function.

        Args:
            - init_params (List[float]): a list of the initial parameters.
        """
        self._init_time = time()
        if init_params is None:
            init_params = [2*pi*random() for _ in range(self.ansatz.number_of_parameters)]

        if len(init_params) < self.num_parameters:
            raise ValueError(
                f"The algorithm needs {self.num_parameters} parameters but only {len(init_params)} were provided!"
            )
        
        out = self.optimizer.optimize(self.evaluate_state, init_param=init_params, **kwargs)

        if self.logger is not None:
           self.logger.close()

        return out


class VQE(VQA):
    def __init__(
        self,
        ansatz: Ansatz,
        cost_function: CostFunction,
        instance: Instance,
        optimizer: Optimizer,
        sampler: Sampler,
        n_shots: int = 1000,
    ):
        super().__init__(ansatz, cost_function, instance, optimizer, sampler, n_shots)


class QAOA(VQA):
    """
    Class representing the QAOA algorithm.

    Args:
        ansatz (Ansatz): The ansatz used in the algorithm.
        backend (Backend): The backend used for circuit execution.
        cost_function (CostFunction): The cost function to be minimized.
        instance (Instance): The problem instance to be solved.
        optimizer (Optimizer): The optimizer used for parameter optimization.
        sampler (Sampler): The sampler used for measurement.
        n_shots (int, optional): The number of measurement shots. Defaults to 1000.

    """

    def __init__(
        self,
        ansatz: Ansatz,
        cost_function: CostFunction,
        instance: Instance,
        optimizer: Optimizer,
        sampler: Sampler,
        n_shots: int = 1000,
        logger: Logger = None
    ):
        super().__init__(ansatz, cost_function, instance, optimizer, sampler, n_shots, logger = logger)

    def solve(self, init_params: list[float] = None, **kwargs):
        """Optimizes a given ansatz according to the specified cost function.

        Args:
            init_params (list[float], optional): A list of the initial parameters. If none are provided, they are
            all initialized to 0.01. Defaults to None.
        """
        if init_params is None:
            init_params = [0.0 for _ in range(self.ansatz.number_of_parameters)]

        elif len(init_params) != self.num_parameters:
            raise ValueError(
                f"The algorithm needs {self.num_parameters} parameters but only {len(init_params)} were provided!"
            )

        self._init_time = time()

        out = self.optimizer.optimize(self.evaluate_state, init_param=init_params, **kwargs)

        if self.logger is not None:
            self.logger.close()

        return out
