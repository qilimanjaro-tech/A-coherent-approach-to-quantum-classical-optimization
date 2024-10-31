import json
import random
from typing import List, Optional, Tuple


class KnapsackInstance:
    """Instance of the Traveling Salesman Problem"""

    def __init__(self, n_items: int, weights: List[int] = None, values: List[int] = None, max_weight_perc: float = 0.5):
        """Initialize a Knapsack instance.

        Args:
            n_items (int): the number of items to be considered in the optimization
            weights (List[int]): the list of weights associated with each item.
            values (List[int]): the list of values associated with each item.
            max_weight_prec (float): the percentage of the total weights that
                                     we can allow in the optimization.
        """
        self.n_items = n_items

        self.weights = weights
        self.values = values
        self.max_w = int(max_weight_perc * sum(self.weights)) if weights is not None else None

    def export_to_file(self, filename: str):
        """Exports the instance to a JSON file.

        Args:
            filename (str): name of the file that will be exported as filename.json
        """
        if self.values is not None:
            data = {"n_items": self.n_items, "weights": self.weights, "values": self.values, "max_weight": self.max_w}

            with open(f"{filename}.json", "w") as file:
                json.dump(data, file)
        else:
            raise ValueError("Instance is not defined")

    def import_from_file(self, filename: str):
        """Imports the instance from a JSON file. It has to include the fields
            'n_items': int
            'weights': List[int]
            'values': List[int]
            'max_weight': int
        Args:
            filename (str): name of the file to import as filename.json
        """
        with open(f"{filename}.json") as file:
            data = json.load(file)
            self.n_items = data["n_items"]
            self.weights = data["weights"]
            self.values = data["values"]
            self.max_w = data["max_weight"]

    def generate_random_instance(
        self,
        max_weight_perc: float = 0.5,
        seed: Optional[int] = None,
        values_range: Tuple[int, int] = (1, 10),
        weights_range: Tuple[int, int] = (1, 10),
    ):
        """Generates a random distances matrix (fully connected) with weights uniformly distributed between
        the bounds specified.

        Args:
            max_weight_perc (float, optional): the percentage of the total weight that you allow in your system.
            The default is 0.5.
            values_range (Tuple[int, int], optional): the lower and upper bounds of the random values used to
            populate the Knapsack item values.
            weights_range (Tuple[int, int], optional): the lower and upper bounds of the random values used to
            populate the Knapsack item weights.
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            random.seed(seed)

        self.weights = [random.randint(weights_range[0], weights_range[1]) for _ in range(self.n_items)]
        self.values = [random.randint(values_range[0], values_range[1]) for _ in range(self.n_items)]
        self.max_w = int(max_weight_perc * sum(self.weights))

    def pprint(self):
        if self.values is None:
            print("-" * 10)
            print("instance is empty!")
            print("-" * 10)
        else:
            print("-" * 10)
            print("weights:", self.weights)
            print("values:", self.values)
            print(f"maximum allowed weight: {self.max_w}")
            print(f"total weight: {sum(self.weights)}")
            print("-" * 10)
