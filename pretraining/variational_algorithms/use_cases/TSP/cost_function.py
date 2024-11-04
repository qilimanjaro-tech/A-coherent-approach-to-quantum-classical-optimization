from typing import List
from variational_algorithms.use_cases.TSP.instances import TSP_Instance
from variational_algorithms.use_cases.cost_function import UseCaseCostFunction
from variational_algorithms.hamiltonians import Z


class TSP_CostFunction(UseCaseCostFunction):
    def __init__(
        self,
        instance: TSP_Instance,
        parameters: List[float] = None,
        encoding: str = "one_hot",
        lagrange_multiplier: float = 10,
    ):
        """
        Args:
            - instance (Instance): the problem instance.
            - sampler (Sampler): A sampler object used to get the expectation values of the circuit output.
            - parameters (List[float]): a list of parameters that can be used in the cost function.
        """
        super().__init__(instance, parameters)

        # TODO: Implement a more general way of passing the lagrange multipliers
        self.lagrange_multiplier = lagrange_multiplier
        for _ in range(instance.m):
            self.define_integer(encoding, instance.m - 1)

        self.encoding = encoding

    @property
    def num_bin_vars(self):
        return self.instance.m * self.vars[0].num_bin_vars

    @property
    def num_constraints(self):
        return 2 * (self.instance.m)

    def __get_path(self, int_sample: List[List[int]]):
        path = []
        if self.instance.start is not None:
            path.append([self.instance.start])
        path.extend(
            [loc if (self.instance.start is None or loc < self.instance.start) else loc + 1 for loc in var]
            for var in int_sample
        )
        if self.instance.loop:
            path.append(path[0])
        return path

    def get_terms(self, int_sample: List[List[int]]):
        path = self.__get_path(int_sample)
        return self.check_cost(path), self.check_constraints(int_sample)

    def check_constraints(self, int_sample: List[List[int]]):
        """Generates a list of the term associated to each constraint for a single sample.

        Returns:
            - List[float]
        """
        return [(sum(time_step.count(i) for time_step in int_sample) - 1) ** 2 for i in range(self.instance.m)]

    def check_cost(self, path: List[int]):
        cost = 0
        for t in range(len(path) - 1):
            for i in path[t]:
                for j in path[t + 1]:
                    cost += self.instance.distances[i][j]
        return cost

    @property
    def constraint_hamiltonian(self):
        if self.encoding != "one_hot":
            raise NotImplementedError()

        m = self.instance.m

        constraint_ham = sum(
            (1 - 0.5 * m + sum(0.5 * Z(i + j * m) for j in range(m)))
            * (1 - 0.5 * m + sum(0.5 * Z(i + j * m) for j in range(m)))
            for i in range(m)
        ) + sum(
            (1 - 0.5 * m + sum(0.5 * Z(i + j * m) for i in range(m)))
            * (1 - 0.5 * m + sum(0.5 * Z(i + j * m) for i in range(m)))
            for j in range(m)
        )

        return constraint_ham

    @property
    def cost_hamiltonian(self):
        if self.encoding != "one_hot":
            raise NotImplementedError()

        ham = 0

        m = self.instance.m
        edge_range = m + 1 if self.instance.start is not None else m

        for edge in [
            (i, j) for i in range(edge_range) for j in range(edge_range) if bool(self.instance.distances[i][j])
        ]:
            weight = self.instance.distances[edge[0]][edge[1]]

            u = edge[0] if self.instance.start is None or edge[0] < self.instance.start else edge[0] - 1
            v = edge[1] if self.instance.start is None or edge[1] < self.instance.start else edge[1] - 1

            for k in range(m - 1):
                if edge[0] == self.instance.start:
                    ham += 0.5 * weight * (1 - Z(v + (k + 1) * m))
                elif edge[1] == self.instance.start:
                    ham += 0.5 * weight * (1 - Z(u + k * m))
                else:
                    ham += 0.25 * weight * (1 - Z(u + k * m)) * (1 - Z(v + (k + 1) * m))

            if self.instance.loop:
                if edge[0] == self.instance.start:
                    ham += 0.5 * weight * (1 - Z(v))
                elif edge[1] == self.instance.start:
                    ham += 0.5 * weight * (1 - Z(u + m * (m - 1)))
                else:
                    ham += 0.25 * weight * (1 - Z(u + m * (m - 1))) * (1 - Z(v))

        return ham

    @property
    def total_hamiltonian(self):
        return self.lagrange_multiplier * self.constraint_hamiltonian + self.cost_hamiltonian
