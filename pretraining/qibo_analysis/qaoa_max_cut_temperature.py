import json
import numpy as np
import networkx as nx
from qaoa_utils import (
    exprectum_hamiltonian_brute_force,
    create_gibbs_state,
    expected_value,
    build_maxcut_hamiltonian,
    build_mixer_hamiltonian,
    max_cut_brute_force,
    sample_state,
    create_gausgibbs_state,
)
from qibo import models


test = 10
test_iter = 5
nodes = [12]
pb_conexion = 0.5
temperature = 4
layers_list = [5]
tol = 0.4

for n_layers in layers_list:

    for n_nodes in nodes:

        for i in range(0, 6):

            delta = 0.1
            iterations = 0
            expected_value_psgibbs_state = 0
            expected_value_gibbs_state = 0

            G = nx.erdos_renyi_graph(n_nodes, pb_conexion)

            dict_values = exprectum_hamiltonian_brute_force(G)

            while (iterations < test_iter):

                print("case: ", i, " delta: ", delta)


                gibbs_state = create_gibbs_state(dict_values, t=temperature)

                expected_value_gibbs_state = expected_value(
                    state=gibbs_state, dict_values=dict_values
                )

                dict_pb_gibbs = sample_state(gibbs_state)

                pb_gibbs = list(dict_pb_gibbs.values())

                entropy_gibbs = -sum(
                    pb_gibbs[i] * np.log(pb_gibbs[i])
                    for i in range(len(pb_gibbs))
                    if pb_gibbs[i] != 0
                )

                pseudo_gibbs_state = create_gausgibbs_state(
                    solutions_dict=dict_values,
                    delta_energy=delta,
                    mu_energy=expected_value_gibbs_state,
                )

                expected_value_psgibbs_state = expected_value(
                    state=pseudo_gibbs_state, dict_values=dict_values
                )

                dict_pb_psgibbs = sample_state(pseudo_gibbs_state)

                pb_psgibbs = list(dict_pb_psgibbs.values())

                entropy_psgibbs = -sum(
                    pb_psgibbs[i] * np.log(pb_psgibbs[i])
                    for i in range(len(pb_psgibbs))
                    if pb_psgibbs[i] != 0
                )

                exact_value = - max_cut_brute_force(G)


                hamiltonian = build_maxcut_hamiltonian(graph=G)
                mixer_hamiltonian = build_mixer_hamiltonian(graph=G)

                qaoa = models.QAOA(hamiltonian=hamiltonian, mixer=mixer_hamiltonian)

                initial_parameters = [0] * 2 * n_layers

                best_energy_gibbs_state, _, _ = qaoa.minimize(
                    initial_p=initial_parameters,
                    method="COBYLA",
                    options={"maxiter": 2000},
                    initial_state=gibbs_state,
                )

                best_energy_pseudo_gibbs_state, _, _ = qaoa.minimize(
                    initial_p=initial_parameters,
                    method="COBYLA",
                    options={"maxiter": 5000},
                    initial_state=pseudo_gibbs_state,
                )

                path_data = (
                    f"logger_data/logger_data_qaoa_performance_gibbs_states_temperature/logger_data_max_cut_qaoa_common_instances_n_{n_nodes}_temperature/"
                    f"cmaes_qaoa_max_cut_n_{n_nodes}_case_{i}_t_{temperature}_"
                    f"layers_{n_layers}_delta_gauss_{round(delta, 3)}.json"
                )

                data_test = {
                    "expected_value_gibbs_state": expected_value_gibbs_state,
                    "expected_value_psgibbs_state": expected_value_psgibbs_state,
                    "energy_qaoa_gibbs_state": best_energy_gibbs_state,
                    "energy_qaoa_psgibbs_state": best_energy_pseudo_gibbs_state,
                    "energy_value_exact": exact_value,
                    "entropy_gibbs_state": entropy_gibbs,
                    "entropy_psgibbs_state": entropy_psgibbs,
                }


                json_datos = json.dumps(data_test, indent=2)

                with open(path_data, 'w') as archivo:
                    archivo.write(json_datos)

                
                delta += 0.2
                iterations += 1