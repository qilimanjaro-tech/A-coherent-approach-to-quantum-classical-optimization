import itertools
import numpy as np
import random
import networkx as nx
from itertools import chain, combinations

from typing import List, Iterable, Set, Optional, Tuple

from qibo.symbols import Z, X
from qibo import hamiltonians

def _powerset(iterable: Iterable[int]) -> List[Set[int]]:
    """_summary_

    Args:
        iterable (Iterable[int]): _description_

    Returns:
        List[Set[int]]: _description_
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def random_uniform_weights(nodes: int, bounds: Tuple[float, float] = (0, 1), seed: Optional[int] = None):
    """Generates a random and symmetric weights matrix with weights uniformly distributed between
    the bounds specified.

    Args:
        bounds (Tuple[float, float], optional): Lower and upper bounds for the weights between each pair
            of locations. Defaults to (0, 1).
        symmetric (bool, optional): Whether the weights between two locations are equal (True) or different
            (False) depending on the direction (weights i->j and j->i).
        seed (int, optional): Seed for the random number generator. Defaults to None.
    """
    weights = None
    
    if seed is not None:
        np.random.seed(seed)
    weights = np.random.uniform(low=bounds[0], high=bounds[1], size=[nodes, nodes])
    #weights = np.random.normal(loc=0.5, scale=1, size=[nodes, nodes])
    for row in range(nodes):
        weights[row][row] = 0
        for col in range(row, nodes):
            weights[col, row] = weights[row, col]
    
    return nx.from_numpy_array(weights)


def max_cut_brute_force(graph):
    """_summary_

    Args:
        graph (_type_): _description_

    Returns:
        _type_: _description_
    """

    n = graph.number_of_nodes()

    max_cut_weight = 0

    for partition in itertools.product([0, 1], repeat=n):
        cut_weight = 0

        for u, v in graph.edges():

            if partition[u] != partition[v]:
                cut_weight += graph[u][v]["weight"] if "weight" in graph[u][v] else 1

        max_cut_weight = max(max_cut_weight, cut_weight)

    return max_cut_weight

def brute_force_advance(matrix):
    """Returns the list of all possile cuts and their associated cut values, sorted by descending cost.

    Raises:
        ValueError: If the weights matrix is not defined

    Returns:
        List[Tuple[set, set, float]]: List of all possible subset/complement cuts and their associated cut values.
    """
    if matrix is None:
        raise ValueError("Weights matrix not defined")
    
    n_nodes = matrix.shape[0]
    
    cut_list = {}
    vertices = set(range(n_nodes))
    
    for subset_tuple in _powerset(vertices):
        
        subset = set(subset_tuple)
        complement = vertices - set(subset)
        
        cut_value = -sum(matrix[i][j] for i in subset for j in complement)
        
        if type(subset)==set():
            cut_list['0'*n_nodes] = float(cut_value)
        
        else: 
            
            subset_list = list(subset)
            integert_to_bin = lambda pos: ''.join(
                '1' if i in pos else '0' for i in range(n_nodes))
            
            state_bin = integert_to_bin(subset_list) 
            
            cut_list[state_bin] = float(cut_value)
        
    cut_list_sorted = dict(sorted(cut_list.items(), key=lambda item: item[1]))
        
    return cut_list_sorted


def build_maxcut_hamiltonian(graph):
    """
    build the MaxCut hamiltonian of a given graph. this takes into account all
    the edges of the graph.

    args:
        graph: graph
            A network graph

    returns:
        The cost hamiltonian of the given graph

    """
    sham = -1 / 2 * sum((1 - Z(i) * Z(j)) for i, j in graph.edges()) 

    return hamiltonians.SymbolicHamiltonian(sham)


def build_mixer_hamiltonian(graph):
    """
    build the mixer hamiltonian for the given graph.

    args:
        graph: graph
            A network graph

    returns:
        The mixer hamiltonian of the given graph

    """
    sham = sum((X(i)) for i in range(graph.number_of_nodes()))

    return hamiltonians.SymbolicHamiltonian(sham)


def create_state(state: list):
    """
    creates a quantum state out of an initial classical prediction.

    args:

    classical_solution: a binary list
        a list of binary values that encode the greedy solution of the problem.

    """
    classical_solution = [int(caracter) for caracter in state]

    nqubits = len(classical_solution)

    initial_state = [0, 1] if classical_solution[0] == 1 else [1, 0]

    for i in range(1, nqubits):
        if classical_solution[i] == 1:
            initial_state = np.kron(initial_state, [0, 1])
        else:
            initial_state = np.kron(initial_state, [1, 0])

    return initial_state


def create_gibbs_state(solutions_dict: dict, t: int = 1) -> np.array:
    """_summary_

    Args:
        solutions_dict (dict): _description_
        t (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """

    psi_state = 0

    for state, energy in solutions_dict.items():

        psi_state += np.exp(-energy / t) * create_state(state)

    psi_state_norm = psi_state / np.linalg.norm(psi_state)

    return psi_state_norm


def create_analitical_psgibbs_state(solutions_gibbs: dict, t: int, alpha) -> np.array:
    """_summary_

    Args:
        solutions_gibbs_dict (dict): _description_
        t (int): _description_
        alpha (float): _description_

    Returns:
        _type_: _description_
    """
    solutions_gibbs_dict = solutions_gibbs.copy()

    psi_psstate = 0

    e_min = min(solutions_gibbs_dict.values())
    n = len(solutions_gibbs_dict)
    m = sum(1 for value in solutions_gibbs_dict.values() if value == e_min)

    for state, energy in solutions_gibbs_dict.items():

        if energy != e_min and energy != 0:

            psi_psstate += np.sqrt(
                np.exp(-2 * energy / t) + (m * alpha * e_min) / ((n - m - 1) * energy)
            ) * create_state(state)

        elif energy == e_min and energy != 0:

            psi_psstate += np.sqrt(np.exp(-2 * energy / t) - alpha) * create_state(
                state
            )

    psi_state_norm = psi_psstate / np.linalg.norm(psi_psstate)

    return psi_state_norm


def create_gausgibbs_state(
    solutions_dict: dict, delta_energy: float, mu_energy: float
) -> np.array:
    """_summary_

    Args:
        solutions_gibbs (dict): _description_
        t (int): _description_
        energy_reference (float): _description_

    Returns:
        np.array: _description_
    """

    psi_psstate = 0

    for state, energy in solutions_dict.items():

        psi_psstate += (
            1/ (2 * delta_energy * np.sqrt(2 * np.pi)) * np.exp(-((energy - mu_energy) ** 2) / (2 * delta_energy**2)) * create_state(state)
        )

    psi_state_norm = psi_psstate / np.linalg.norm(psi_psstate)

    return psi_state_norm


def create_local_state(solutions_dict: dict, reference_energy: float, left: bool = True):
    """_summary_

    Args:
        solutions_dict (dict): _description_
        type_state (middle&quot;, &quot;low&quot;], optional): _description_. Defaults to "middle".

    Returns:
        _type_: _description_
    """

    one_state_str = select_state(
        dict_states=solutions_dict, reference_energy=reference_energy, left = left
    )

    one_state = create_state(state=one_state_str)

    return one_state

def create_random_state(solutions_dict: dict, max_energy: float):
    """_summary_

    Args:
        solutions_dict (dict): _description_
        max_energy (float): _description_

    Returns:
        _type_: _description_
    """
    
    dict_state_low = dict([(state, energy) for state, energy in solutions_dict.items() if energy <= max_energy])
    
    psi_psstate = 0
    psi_state_norm = [np.nan]
    
    while np.isnan(psi_state_norm[0]) != False:
    
        list_zeros = [0.0]* random.choice([1, 5, 10, 20, 25, 35, len(dict_state_low)])
        list_pb = list_zeros + [1.0]

        for state, _ in dict_state_low.items():

            psi_psstate += random.choice(list_pb) * create_state(state)

        psi_state_norm = psi_psstate / np.linalg.norm(psi_psstate)
    
    return psi_state_norm


def select_state(dict_states, reference_energy, left = True):
    """_summary_

    Args:
        dict_states (_type_): _description_
        reference_energy (_type_): _description_

    Returns:
        _type_: _description_
    """
    if left:
        
        dict_state_low = [
            (state, energy)
            for state, energy in dict_states.items()
            if energy < reference_energy
        ]

        state, _ = max(dict_state_low, key=lambda x: x[1])
    
    else: 
        
        dict_state = [
            (state, energy)
            for state, energy in dict_states.items()
            if energy > reference_energy
        ]
        
        state, _ = min(dict_state, key=lambda x: x[1])

    return state


def exprectum_hamiltonian(grahp) -> dict:
    """_summary_

    Args:
        ham (_type_): _description_
        nqubits (_type_): _description_
    """

    ham = build_maxcut_hamiltonian(grahp)
    nqubits = grahp.number_of_nodes()

    h = ham.calculate_dense()

    diag_vec = np.diag(h.matrix)

    ev = sorted(zip(diag_vec, [i for i in range(len(diag_vec))]), key=lambda x: x[0])
    diag, ind = zip(*ev)

    states = []

    states = {}

    for j in range(len(diag)):
        a = "{0:0{bits}b}".format(ind[j], bits=nqubits)
        states[a] = np.real(diag[j])

    return states

def exprectum_hamiltonian_brute_force(grahp):
    """_summary_

    Args:
        grahp (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    adjacency_matrix = nx.adjacency_matrix(grahp)
    adjacency_matrix_np = adjacency_matrix.toarray()
    
    results = brute_force_advance(adjacency_matrix_np)
    
    return results


def expected_value(state: np.array, dict_values: dict) -> float:
    """_summary_

    Args:
        state (np.array): _description_
        dict_values (dict): _description_

    Returns:
        float: _description_
    """

    dict_samples = sample_state(state)

    expected = 0

    for clave in dict_samples.keys() & dict_values.keys():
        expected += dict_values[clave] * dict_samples[clave]

    return expected


def sample_state(state: np.array) -> dict:
    """_summary_

    Args:
        state (str): _description_

    Returns:
        _type_: _description_
    """

    probabilities = np.abs(state) ** 2
    nqubits = int(np.log2(len(probabilities)))
    probabilities = zip(probabilities, [i for i in range(len(probabilities))])
    probabilities = sorted(probabilities, key=lambda x: x[0], reverse=True)
    probabilities = {
        "{0:0{bits}b}".format(s, bits=nqubits): p for p, s in probabilities
    }

    return probabilities

def create_random_state_no_energy(solutions_dict: dict):
    """_summary_

    Args:
        solutions_dict (dict): _description_

    Returns:
        _type_: _description_
    """
    
    psi_psstate = 0
    psi_state_norm = [np.nan]
    
    while np.isnan(psi_state_norm[0]) != False:
    
        prob_accept_element = np.random.uniform(0.01,1)
        n_items = len(solutions_dict.items())

        random_state = np.random.choice(n_items)
        target_state = list(solutions_dict.keys())[random_state]
        for state, _ in solutions_dict.items():
            #list_pb = list_zeros + [np.random.uniform([1,10])]
            if state == target_state:
                psi_psstate +=  np.random.choice([1,1000]) * create_state(state)
                continue
            if random.uniform(0,1)<prob_accept_element:
                psi_psstate += np.random.choice([1,1000]) * create_state(state)

        psi_state_norm = psi_psstate / np.linalg.norm(psi_psstate)
    
    return psi_state_norm