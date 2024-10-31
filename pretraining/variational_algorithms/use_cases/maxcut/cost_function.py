from variational_algorithms.cost_function import CostFunction
from variational_algorithms.use_cases.cost_function import UseCaseCostFunction
from variational_algorithms.hamiltonians import Z
from variational_algorithms.use_cases.maxcut.instances import MaxCut_Instance
from typing import List


class MaxCut_CostFunction(UseCaseCostFunction):
    def __init__(self, instance: MaxCut_Instance, parameters: List[float] = None):
        super().__init__(instance, parameters)

        for _ in range(instance.n_nodes):
            self.define_binary()

        self.lagrange_multiplier = 1

    @property
    def num_bin_vars(self):
        return self.instance.n_nodes

    @property
    def num_constraints(self):
        return 0

    def _get_set1(self, bin_sample: List[List[int]]):
        return [index for index, value in enumerate(bin_sample) if value == [1]]

    def check_constraints(self, *args):
        return []

    def check_cost(self, set1: List[int]):
        complement = list(set(range(self.instance.n_nodes)) - set(set1))
        cost = -sum(self.instance.weights[i][j] for i in set1 for j in complement)

        return cost

    def check_cost_binary(self, set: str):

        list_integer  = []

        for i, value in enumerate(set):
            if value == '1':
                list_integer.append(i)

        return self.check_cost(list_integer)

    def get_terms(self, bin_sample: List[int]):
        set1 = self._get_set1(bin_sample)
        return self.check_cost(set1), self.check_constraints(set1)

    @property
    def constraint_hamiltonian(self):
        return 0

    @property
    def cost_hamiltonian(self):
        ham = 0
        for edge in [
            (i, j) for i in range(self.instance.n_nodes) for j in range(i) if bool(self.instance.weights[i][j])
        ]:
            ham = ham + (-0.5) * self.instance.weights[edge[0]][edge[1]] * (1 - Z(edge[0]) * Z(edge[1]))

        return ham

    @property
    def total_hamiltonian(self):
        return self.cost_hamiltonian + self.lagrange_multiplier * self.constraint_hamiltonian
