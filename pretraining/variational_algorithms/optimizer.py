from abc import abstractmethod
from copy import copy, deepcopy
from typing import Callable, Dict, List, Optional, Literal, Tuple

import warnings

import cma
import numpy as np
from scipy import optimize as sci_opt
from variational_algorithms.cost_function import CostFunction

np.set_printoptions(threshold=10)


class Optimizer:
    def __init__(self):
        self.optimal_parameters = None

    @abstractmethod
    def parse_cost(self, cost_function: CostFunction):
        """Parses the necessary information from the cost function to evaluate the cost.

        Args:
            - cost_function (CostFunction): a cost function object containing methods to evaluate the objective,
            constraints, and encoding costs.

        Returns:
            - the input expected by the optimizer.
        """

    @abstractmethod
    def optimize(self, func: Callable, init_param: List[float], **kwargs):
        """Optimizes the parameters.

        Args:
            - func (Callable): a function that evaluates the optimization step.
            - init_params (Lit[float]): a list of the initial parameters.
            - **kwargs: optimizer specific parameters.
        """

    @abstractmethod
    def get_optimal_parameters(self) -> List[float]:
        """Gets the optimal set of parameters after the optimization process.

        Returns:
            - a list of optimal parameters.
        """


class ScipyOptimizer(Optimizer):
    def __init__(self, max_iter=None):

        self.max_iter = max_iter

    def to_dict(self):
        """Generates a dictionary containing all the information of the inistance.

        Returns:
            Dict: Dictionary with the information to recreate the class instance.
        """
        return {"_name": self.__class__.__name__, "_optimal_parameters": self.optimal_parameters}

    def parse_cost(self, cost_function: CostFunction):
        """Parses the necessary information from the cost function to evaluate the cost.

        Args:
            - cost_function (CostFunction): a cost function object containing methods to evaluate the objective, constraints, and encoding costs.

        Returns:
            - the input expected by the optimizer.
        """
        return (
            cost_function.lagrange_multiplier * (cost_function.constraints_cost() + cost_function.encoding_cost())
            + cost_function.objective_cost()
        )

    def optimize(self, func: Callable, init_param: List[float], max_iter=None, **kwargs):
        """Optimizes the parameters.

        Args:
            - func (Callable): a function that evaluates the optimization step.
            - init_params (Lit[float]): a list of the initial parameters.
            - **kwargs: optimizer specific parameters.
        """

        method = kwargs.pop("method", "Powell")
        
        options = None
        if max_iter:
            self.max_iter = max_iter
        if self.max_iter:
            if method == "TNC":
                options = {"maxfun": self.max_iter}
            else:
                options = {"maxiter": self.max_iter}           

        res = sci_opt.minimize(func, method=method, x0=init_param, options=options, **kwargs)

        self.optimal_parameters = res.x

        return res

    def get_optimal_parameters(self) -> List[float]:
        """Gets the optimal set of parameters after the optimization process.

        Returns:
            - a list of optimal parameters.
        """
        return self.optimal_parameters.tolist()


class CMA_ES(Optimizer):
    def __init__(self, max_eval=None):
        self.max_eval = max_eval
        
    def to_dict(self):
        """Generates a dictionary containing all the information of the inistance.

        Returns:
            Dict: Dictionary with the information to recreate the class instance.
        """
        return {
            "_name": self.__class__.__name__,
            "_optimal_parameters": self.optimal_parameters,
        }

    def parse_cost(self, cost_function: CostFunction):
        """Parses the necessary information from the cost function to evaluate the cost.

        Args:
            - cost_function (CostFunction): a cost function object containing methods to evaluate the objective, constraints, and encoding costs.

        Returns:
            - the input expected by the optimizer.
        """
        return (
            cost_function.lagrange_multiplier * (cost_function.constraints_cost() + cost_function.encoding_cost())
            + cost_function.objective_cost()
        )

    def optimize(self, func: Callable, init_param: List[float], max_eval=None, **kwargs):
        """Optimizes the parameters.

        Args:
            - func (Callable): a function that evaluates the optimization step.
            - init_params (Lit[float]): a list of the initial parameters.
            - **kwargs: optimizer specific parameters.
        """

        init_dev = kwargs.pop("init_dev", 1e-2)

        options = {"verb_log": 0, **kwargs.pop("options", {})}

        if max_eval:
            self.max_eval = max_eval
        if self.max_eval:
            options["maxfevals"] = self.max_eval

        xopt, es = cma.fmin2(func, init_param, init_dev, options)

        self.optimal_parameters = xopt

        return es.result

    def get_optimal_parameters(self) -> List[float]:
        """Gets the optimal set of parameters after the optimization process.

        Returns:
            - a list of optimal parameters.
        """
        return self.optimal_parameters.tolist()
