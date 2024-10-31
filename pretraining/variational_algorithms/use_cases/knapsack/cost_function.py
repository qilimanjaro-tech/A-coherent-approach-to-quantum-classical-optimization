from typing import List

import numpy as np
from variational_algorithms.cost_function import CostFunction
from variational_algorithms.use_cases.instance import Instance


class KnapsackCostFunction(CostFunction):
    """Represents a cost function for the Knapsack problem.

    Args:
        instance: The instance of the Knapsack problem.
        parameters: The list of parameters for the cost function. Default is an empty list.
        unbalanced_penalization: A boolean indicating whether unbalanced penalization is enabled.
                                 Default is False.
        lagrange_multiplier: The Lagrange multiplier for the cost function. Default is 10.0.

    Attributes:
        unbalanced_penalization: A boolean indicating whether unbalanced penalization is enabled.
        lagrange_multiplier: The Lagrange multiplier for the cost function.

    Properties:
        num_bin_vars: The number of binary variables in the cost function.
        num_constraints: The number of constraints in the cost function.

    Methods:
        number_of_parameters: Returns the number of parameters for the cost function.
        process_sample: Processes a sample of variable values.
        check_constraints: Generates a list of the values associated
                           with each constraint for a single sample.
        check_slack_constraints: Generates a list of the term associated with
                                 each constraint for a single sample using slack variables.
        check_unbalanced_penalization_constraints: Generates a list of the
                        term associated with each constraint for a single
                        sample using unbalanced penalization.
        check_cost: Calculates the cost of a sample.
        get_terms: Returns the cost and constraint terms for a sample.
    """

    def __init__(
        self,
        instance: Instance,
        parameters: List[float] = None,
        unbalanced_penalization: bool = False,
        lagrange_multiplier: float = 10.0,
        optimize_cost_function: bool = False,
    ):
        super().__init__(instance, parameters)

        self.unbalanced_penalization = unbalanced_penalization
        self.lagrange_multiplier = lagrange_multiplier
        self.optimize_cost_function = optimize_cost_function

        for i in range(instance.n_items):
            self.define_binary(f"x{i}")

        if not self.unbalanced_penalization:
            for i in range(int(np.ceil(np.log2(self.instance.max_w))) + 1):
                self.define_binary(f"s{i}")

    @property
    def num_bin_vars(self) -> int:
        """Returns the number of binary variables in the CostFunction.

        Returns:
            int: The number of binary variables.

        """
        if self.unbalanced_penalization:
            return self.instance.n_items
        return self.instance.n_items + int(np.ceil(np.log2(self.instance.max_w + 1)))

    @property
    def num_constraints(self) -> int:
        """Returns the number of constraints in the CostFunction.

        Returns:
            int: The number of constraints.

        """
        return 1

    def number_of_parameters(self):
        return 2 if self.optimize_cost_function else 0

    def process_sample(self, vars_values):
        """Processes a sample of variable values.

        Args:
            vars_values: The variable values of the sample.

        Returns:
            The processed sample.

        """
        return vars_values

    def check_constraints(self, state: List[List[int]]) -> List[float]:
        """Generates a list of the term associated to each constraint for a single sample.

        Args:
            state: a single state from the sample list.

        Returns:
            - List[float]
        """
        return [
            self.lagrange_multiplier * c
            for c in (
                self.check_unbalanced_penalization_constraints(state)
                if self.unbalanced_penalization
                else self.check_slack_constraints(state)
            )
        ]

    def calculate_sample_weight(self, state: List[int]) -> float:
        """Calculates the weight of a sample.

        Args:
            state: The state of the sample.

        Returns:
            float: The weight of the sample.

        """
        return sum(self.instance.weights[i] * state[i] for i in range(self.instance.n_items))

    def check_slack_constraints(self, state: List[int]):
        """Generates a list of the term associated with each slack constraint for a single sample.

        Args:
            state: The state of the sample.

        Returns:
            List[float]: The list of slack constraint values.

        Raises:
            AssertionError: If the length of the state is less than the required length.

        """
        num_items = self.instance.n_items
        m = int(np.ceil(np.log2(self.instance.max_w + 1)))

        assert len(state) >= num_items + m

        y = [state[num_items + i] for i in range((m))]
        x = [state[i] for i in range(self.instance.n_items)]

        const = (
            sum(2**i * y[i] for i in range(m-1))
            + (self.instance.max_w + 1 - 2**(m-1)) * y[m-1]
            + self.calculate_sample_weight(x) - self.instance.max_w
        ) ** 2
        return [const]

    def check_unbalanced_penalization_constraints(self, state: List[List[int]]):
        """Generates a list of the term associated with each unbalanced
           penalization constraint for a single sample.

        Args:
            state: The state of the sample.

        Returns:
            List[float]: The list of unbalanced penalization constraint terms.

        """

        h = self.instance.max_w - self.calculate_sample_weight(state)

        if self.optimize_cost_function:
            const = -self.parameters[0] * h + self.parameters[1] * (h**2)
        else:
            const = -h + (h**2)
            # const = -0.9603 * h + 0.0371 * (h**2)

        return [const]

    def check_cost(self, state: List[int]):
        """Calculates the cost of a sample.

        Args:
            state: The state of the sample.

        Returns:
            float: The cost of the sample.

        """
        return sum(-value * state[i] for i, value in enumerate(self.instance.values))

    def get_terms(self, int_sample: List[int]):
        """Returns the cost and constraint terms for a sample.

        Args:
            int_sample: The integer representation of the sample.

        Returns:
            Tuple[float, List[float]]: The cost and constraint terms.

        """
        if isinstance(int_sample, str):
            sample = [int(i) for i in int_sample]
        else:
            sample = int_sample
        return self.check_cost(sample), self.check_constraints(sample)

    def get_total_cost(self, int_sample):
        cost, [constr] = self.get_terms(int_sample)
        return cost + constr

    def calculate_sample_output(self, state: List[List[int]]) -> (float, float):
        """Calculates the output values for a sample.

        Args:
            state: The state of the sample.

        Returns:
            Tuple[float, float]: The calculated value and weight of the sample.

        """
        sample = [int(i) for i in state]
        parsed_sample = self.parse_sample(sample)
        integer_sample = [var.get_value(parsed_sample[i]) for i, var in enumerate(self.vars)]

        value = self.check_cost(integer_sample)
        weight = self.calculate_sample_weight(integer_sample)

        return value, weight
