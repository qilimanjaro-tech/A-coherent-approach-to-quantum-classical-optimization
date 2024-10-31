from variational_algorithms.use_cases.instance import Instance
from itertools import chain, combinations
import json
import numpy as np
import random
from typing import Optional, Tuple, List, Iterable, Set


class MaxCut_Instance(Instance):
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.weights = None

        self.best_known_cost = None
        self.best_known_sol = None

    @property
    def upper_bound(self):
        return 0

    @property
    def lower_bound(self):
        raise NotImplementedError()

    def export_to_file(self, filename: str, id: int=None, append:bool=True):
        """Exports the weights matrix to a JSON file.

        Args:
            filename (str): name of the file that will be exported as filename.json
        """
        if self.weights is not None:
            new_data = {"id": id, "n_nodes": self.n_nodes, "weights": [[j.item() if isinstance(j, np.generic) else j
                                                                        for j in i] for i in self.weights],
                        "best_known": (self.best_known_cost, self.best_known_sol)}
            if append:
                try:
                    with open(f"{filename}.json", "r") as file:
                        data = json.load(file)
                except:
                    data = {}

                if 'instances' in data:
                    data['instances'].extend([new_data])
                elif 'id' in data:
                    data = {'instances': [data, new_data]}
                else:
                    data = {'instances': [new_data]}
            else:
                data = new_data

            with open(f"{filename}.json", "w") as file:
                json.dump(data, file)

            if append:
                with open(f"{filename}.json", "r+") as file:
                    raw_data = file.read()
                    new_data = raw_data.replace(' {', '\n    {').replace('[{', '[\n    {')

                    file.seek(0)
                    file.write(new_data)
                    file.truncate()

        else:
            raise ValueError("weights matrix not defined")

    def import_from_file(self, filename: str, id:int=None):
        """Imports the weights matrix from a JSON file. It has to include the fields
            'n_nodes': int
            'weights': List[List[float]]

        Args:
            filename (str): name of the file to import as filename.json
        """
        with open(f"{filename}.json") as file:
            data = json.load(file)
            if 'instances' not in data:
                self.n_nodes = data["n_nodes"]
                self.weights = data["weights"]
            else:
                instance = data['instances'][-1]
                if id is not None:
                    for inst in data['instances'][::-1]:
                        if inst['id'] == id:
                            instance = inst
                            break
                self.n_nodes = instance["n_nodes"]
                self.weights = instance["weights"]
                self.best_known_cost, self.best_known_sol = instance["best_known"]

    # TODO: Add methods to generate instances without weights and with different graph connectivities
    def random_uniform_weights(
        self, bounds: Tuple[float, float] = (0, 1), symmetric: bool = False, seed: Optional[int] = None
    ):
        """Generates a random and symmetric weights matrix with weights uniformly distributed between
        the bounds specified.

        Args:
            bounds (Tuple[float, float], optional): Lower and upper bounds for the weights between each pair
                of locations. Defaults to (0, 1).
            symmetric (bool, optional): Whether the weights between two locations are equal (True) or different
                (False) depending on the direction (weights i->j and j->i).
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)
        self.weights = np.random.uniform(low=bounds[0], high=bounds[1], size=[self.n_nodes, self.n_nodes])
        for row in range(self.n_nodes):
            self.weights[row][row] = 0
            for col in range(row, self.n_nodes):
                self.weights[col, row] = self.weights[row, col]
                
    def random_erdos_renyi(self, pb_conexion: Optional[float] = 0.5, seed: Optional[int] = None):
        
        if seed is not None:
            random.seed(seed)

        if pb_conexion < 0 or pb_conexion > 1:
            raise ValueError("Probability of connection must be between 0 and 1")
        
        self.weights = np.zeros((self.n_nodes, self.n_nodes), dtype=int)

        for i in range(self.n_nodes):
            for j in range(i + 1, self.n_nodes):
                value = random.choices([0, 1], weights=[1 - pb_conexion, pb_conexion])[0]
                self.weights[i, j] = value
                self.weights[j, i] = value

    def _powerset(self, iterable: Iterable[int]) -> List[Set[int]]:
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))

    def brute_force(self):
        """Returns the list of all possile cuts and their associated cut values, sorted by descending cost.

        Raises:
            ValueError: If the weights matrix is not defined

        Returns:
            List[Tuple[set, set, float]]: List of all possible subset/complement cuts and their associated cut values.
        """
        if self.weights is None:
            raise ValueError("Weights matrix not defined")
        cut_list = []
        vertices = set(range(self.n_nodes))
        for subset_tuple in self._powerset(vertices):
            subset = set(subset_tuple)
            complement = vertices - set(subset)
            cut_value = -sum(self.weights[i][j] for i in subset for j in complement)
            cut_list.append((subset, complement, cut_value))

        cut_list.sort(key=lambda x: x[2], reverse=False)

        return cut_list
