from copy import deepcopy
from typing import List, Dict
from variational_algorithms.cost_function import CostFunction
from variational_algorithms.use_cases.instance import Instance


class UseCaseCostFunction(CostFunction):
    def __init__(self, instance: Instance, parameters: List[float] = []):
        super().__init__(instance, parameters)
