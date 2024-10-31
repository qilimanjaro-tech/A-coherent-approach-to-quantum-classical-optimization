from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from variational_algorithms.use_cases.instance import Instance


class CostFunction:
    """Base CostFunction class

    Args:
        - instance (Instance): the problem instance.
        - sampler (Sampler): A sampler object used to get the expectation values of the circuit output.
        - parameters (List[float]): a list of parameters that can be used in the cost function.
    """

    def __init__(self, instance: Instance, parameters: List[float] = None):
        self.instance = instance
        self.parameters = [] if parameters is None else parameters
        self.vars = []
        self.lagrange_multiplier = None

        self.probabilities = None
        self.encoding_terms = None
        self.constraints_terms = None
        self.cost = None
        self.n_unique_samples = None

    def number_of_parameters(self):
        """A static method that returns the number of parameters needed by the cost function.

        Args:
            - instance (Instance): the problem instance.
        """
        return 0

    def set_parameters(self, parameters: List[float]):
        """Updates the value of the parameters.

        Args:
           - parameters (List[float]): a list of parameters that can be used in the cost function.
        """
        self.parameters = parameters

    def set_samples(self, probabilities: Dict[Tuple[int], int]):
        """Updates the value of the sampler.

        Args:
            - sampler (Sampler): A sampler object used to get the expectation values of the circuit output.
        """
        self.probabilities = probabilities
        self.encoding_terms = []
        self.constraints_terms = []
        self.cost = []
        self.n_unique_samples = len(probabilities)

        for sample, prob in probabilities.items():
            sample_list = list(sample)
            parsed_sample = self.parse_sample(sample_list)
            integer_sample = [var.get_value(parsed_sample[i]) for i, var in enumerate(self.vars)]

            self.encoding_terms.append(
                ([var.encoding_constraint(parsed_sample[i]) for i, var in enumerate(self.vars)], prob)
            )
            cost_term, constr_term = self.get_terms(integer_sample)
            self.constraints_terms.append((constr_term, prob))
            self.cost.append((cost_term, prob))

    def get_terms(self, *args):
        return NotImplementedError()

    def constraints_cost(self, lagrange_multiplier: Dict[str, float] = None) -> float:
        """Calculates the average cost of the constraints for the circuit output.

        List of constraints:
            1- ...

        Args:
            - lagrange_multiplier (Dict[str, float]): a dictionary containing the constraints id and a lagrange
            multiplier to be multiplied by the constraints
        """
        return sum(sum(constr_terms) * prob for (constr_terms, prob) in self.constraints_terms)

    def encoding_cost(self, lagrange_multiplier: Dict[str, float] = None) -> float:
        """Calculates the average cost of the encoding constraints for the circuit output.

        List of encoding constraints:
            1- ...

        Args:
            - lagrange_multiplier (Dict[str, float]): a dictionary containing the constraints id and a lagrange
            multiplier to be multiplied by the constraints
        """

        return sum(sum(enc_terms) * prob for (enc_terms, prob) in self.encoding_terms)

    def objective_cost(self) -> float:
        "Calculates the cost of the objective for the circuit output."
        return sum(sample_cost * prob for (sample_cost, prob) in self.cost)

    def number_of_broken_constraints(self) -> int:
        "Returns the number of broken constraints."
        raise NotImplementedError()

    def define_binary(self, label=None):
        self.vars.append(BinaryVar(label))

    def define_integer(self, encoding="one_hot", upper_bound=1, lower_bound=0, label=None):
        encoding_class = IntegerVar.implemented_encodings()[encoding]
        integer_var = encoding_class(upper_bound, lower_bound, label)
        self.vars.append(integer_var)

    def parse_sample(self, sample):
        # Sampler returned all variables in the defined encoding
        if len(sample) == sum(var.num_bin_vars for var in self.vars):
            return [[int(sample.pop(0)) for _ in range(var.num_bin_vars)] for var in self.vars]
        # Sampler returned all variables as integer variables
        elif len(sample) == len(self.vars):
            return [[int(sample.pop(0))] for _ in self.vars]
        else:
            raise ValueError("Sample could not be resolved.")

    def cost_compute(self) -> float:
        "Calculates the cost function value."

        return self.lagrange_multiplier * (self.constraints_cost() + self.encoding_cost()) + self.objective_cost()


class IntegerVar:
    def __init__(self, upper_bound, lower_bound, label):
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.label = label

    @staticmethod
    def implemented_encodings():
        return {"integer": IntegerVar, "one_hot": OneHotVar, "domain_wall": DomainWallVar, "HOBO": HOBOVar}

    @property
    def num_bin_vars(self):
        return 1

    def encoding_constraint(self, bin_vars):
        """Returns the cost of a one-hot encoding constraint for a given set of variables.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: value of the constraint term
        """
        return 1 if self.get_value(bin_vars)[0] > self.upper_bound else 0

    def get_value(self, value):
        return [integer + self.lower_bound for integer in value]


class BinaryVar(IntegerVar):
    def __init__(self, label):
        super().__init__(1, 0, label)

    @property
    def num_bin_vars(self):
        return 1

    def encoding_constraint(self, bin_vars):
        """Returns the cost of a one-hot encoding constraint for a given set of variables.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: value of the constraint term
        """
        return 0

    def get_value(self, bin_vars):
        """Returns the value that a set of variables represent in a one-hot encoding.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: Value of the integer variable
        """
        return bin_vars


class OneHotVar(IntegerVar):
    @property
    def num_bin_vars(self) -> int:
        return int(self.upper_bound - self.lower_bound + 1)

    def encoding_constraint(self, bin_vars):
        """Returns the cost of a one-hot encoding constraint for a given set of variables.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: value of the constraint term
        """
        return (bin_vars.count(1) - 1) ** 2

    def get_value(self, bin_vars):
        """Returns the value that a set of variables represent in a one-hot encoding.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: Value of the integer variable
        """
        return [index + self.lower_bound for index, value in enumerate(bin_vars) if value]


class DomainWallVar(IntegerVar):
    @property
    def num_bin_vars(self) -> int:
        return int(self.upper_bound - self.lower_bound)

    def encoding_constraint(self, bin_vars):
        """Returns the cost of a domain-wall encoding constraint for a given set of variables.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: value of the constraint term
        """
        return sum(bin_vars[i + 1] * (1 - bin_vars[i]) for i in range(len(bin_vars) - 1))

    def get_value(self, bin_vars):
        """Returns the value that a set of variables represent in a domain-wall encoding.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: Value of the integer variable
        """
        return ([self.lower_bound] if bin_vars[0] == 0 else []) + [
            self.lower_bound + index + 1
            for index, value in enumerate(bin_vars)
            if (value == 1 and (index == len(bin_vars) - 1 or bin_vars[index + 1] == 0))
        ]


class HOBOVar(IntegerVar):
    @property
    def num_bin_vars(self) -> int:
        return int(np.ceil(np.log2(self.upper_bound - self.lower_bound + 1)))

    def encoding_constraint(self, bin_vars):
        """Returns the cost of a HOBO encoding constraint for a given set of variables.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: value of the constraint term
        """
        return 1 if len(self.get_value(bin_vars)) == 0 else 0

    def get_value(self, bin_vars):
        """Returns the value that a set of variables represent in a HOBO encoding.

        Args:
            bin_vars (List[int]): Values for the binary variables
        Returns:
            int: Value of the integer variable
        """
        value = sum(2**i * var for i, var in enumerate(bin_vars[::-1])) + self.lower_bound
        return [value] if value <= self.upper_bound else []
