from variational_algorithms.cost_function import CostFunction
from instances import UseCaseInstance
from typing import List, Dict

class UseCaseCostFunction(CostFunction):
    def __init__(self, instance: UseCaseInstance, parameters: List[float] = []):
        """
        Args:
            - instance (Instance): the problem instance.
            - sampler (Sampler): A sampler object used to get the expectation values of the circuit output.
            - parameters (List[float]): a list of parameters that can be used in the cost function. 
        """
        super().__init__()
        pass

    def list_constraints(self):
        """ Generates a list of the term associated to each constraint.
        
        Returns:
            - List[float]
        """
        raise NotImplementedError()
