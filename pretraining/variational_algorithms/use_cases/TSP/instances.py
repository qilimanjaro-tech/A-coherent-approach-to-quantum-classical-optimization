from copy import deepcopy
from typing import Optional, Tuple, List, Dict
import json
import numpy as np
from variational_algorithms.use_cases.instance import Instance


class TSP_Instance(Instance):
    """Instance of the Traveling Salesman Problem"""

    def __init__(self, n_nodes: int, start: Optional[int] = None, loop: bool = True):
        self.n_nodes = n_nodes
        self.start = start
        self.loop = loop
        self.m = self.n_nodes if start is None else self.n_nodes - 1
        self.distances = None

    @property
    def upper_bound(self):
        return max(self.distances.flatten()) * self.m

    @property
    def lower_bound(self):
        raise NotImplementedError()

    def export_to_file(self, filename: str):
        """Exports the distances matrix to a JSON file.

        Args:
            filename (str): name of the file that will be exported as filename.json
        """
        if self.distances is not None:
            data = {"n_nodes": self.n_nodes, "distances": [list(i) for i in self.distances]}
            with open(f"{filename}.json", "w") as file:
                json.dump(data, file)
        else:
            raise ValueError("Distances matrix not defined")

    def import_from_file(self, filename: str):
        """Imports the distances matrix from a JSON file. It has to include the fields
            'n_nodes': int
            'distances': List[List[float]]

        Args:
            filename (str): name of the file to import as filename.json
        """
        with open(f"{filename}.json") as file:
            data = json.load(file)
            self.n_nodes = data["n_nodes"]
            self.distances = data["distances"]

    def random_uniform_distances(
        self, bounds: Tuple[float, float] = (0, 1), symmetric: bool = False, seed: Optional[int] = None
    ):
        """Generates a random distances matrix (fully connected) with weights uniformly distributed between
        the bounds specified.

        Args:
            bounds (Tuple[float, float], optional): Lower and upper bounds for the distances between each pair
                of locations. Defaults to (0, 1).
            symmetric (bool, optional): Whether the distances between two locations are equal (True) or different
                (False) depending on the direction (distances i->j and j->i).
            seed (int, optional): Seed for the random number generator. Defaults to None.
        """
        if seed is not None:
            np.random.seed(seed)
        self.distances = np.random.uniform(low=bounds[0], high=bounds[1], size=[self.n_nodes, self.n_nodes])
        for row in range(self.n_nodes):
            self.distances[row][row] = 0
            if symmetric:
                for col in range(row, self.n_nodes):
                    self.distances[col, row] = self.distances[row, col]

    def __brute_force_step(self, path: List, cost: float) -> List[Tuple[List, float]]:
        remaining = set(range(self.n_nodes)) - set(path)
        if len(remaining) == 0:
            if self.loop:
                cost += self.distances[path[-1]][path[0]]
                path.append(path[0])
            return [(path, cost)]

        paths_list = []
        for location in remaining:
            new_paths_list = self.__brute_force_step(
                path + [location], cost + self.distances[path[-1]][location] if len(path) > 0 else cost
            )

            paths_list.extend(new_paths_list)

        return paths_list

    def brute_force(self) -> List[Tuple[List, float]]:
        """Returns the set of possible paths found by brute force and sorted by cost.

        Args:
            start (int, optional): Location from which the path starts,
                if None the first location is not fixed. Defaults to None.
            loop (bool, optional): Whether the path must end and the starting location or not. Defaults to True.

        Returns:
            paths_list (List[Tuple[path(List), cost(float)]]): List of paths that fulfill the TSP constraints,
                sorted by the total cost. Each element in the list corresponds to a tuple, with the first element
                being the list of locations conforming the path and the second element being the cost associated.
        """
        if self.distances is None:
            raise ValueError("Distances matrix not defined")
        initial = [] if self.start is None else [self.start]
        paths_list = self.__brute_force_step(initial, 0)
        paths_list.sort(key=lambda tup: tup[1], reverse=False)

        return paths_list
