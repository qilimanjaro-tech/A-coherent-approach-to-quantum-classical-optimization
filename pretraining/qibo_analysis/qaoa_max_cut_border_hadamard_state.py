import json
import networkx as nx
from qaoa_utils import (
    create_gibbs_state,
    expected_value,
    build_maxcut_hamiltonian,
    build_mixer_hamiltonian,
    exprectum_hamiltonian_brute_force,
)
from qibo import models


test = 30
pb_conexion = 0.7
temperatures = [0.5, 1, 2, 3, 4, 5, 6]
n_layers_list = [3,5]

for temperature in temperatures:

    for z in range(len(n_layers_list)):
        
        n_layers = n_layers_list[z]

        for n_nodes in range(14, 13, -1):
            
            i = 0
            
            while i < test:

                print('-----------------------------------------')
                print("temperature: ", temperature, " case: ", i,)
                print('-----------------------------------------')

                G = nx.erdos_renyi_graph(n_nodes, pb_conexion)

                dict_values = exprectum_hamiltonian_brute_force(G)
                
                print('-----------------------------')
                print('Gibbs state create')
                print('-----------------------------')

                gibbs_state = create_gibbs_state(dict_values, t=temperature)
                
                print('-----------------------------')
                print('Gibbs state expected')
                print('-----------------------------')

                expected_value_gibbs_state = expected_value(
                    state=gibbs_state, dict_values=dict_values
                )
                
                print('-----------------------------')
                print('Hadamard state')
                print('-----------------------------')
                
                hadamard_state = create_gibbs_state(dict_values, t=10**6)

                expected_value_hadamard_state = expected_value(
                    state=hadamard_state, dict_values=dict_values
                )
                
                print('-----------------------------')
                print('Simulations')
                print('-----------------------------')

                exact_value = min(dict_values.values())

                hamiltonian = build_maxcut_hamiltonian(graph=G)
                mixer_hamiltonian = build_mixer_hamiltonian(graph=G)

                qaoa = models.QAOA(hamiltonian=hamiltonian, mixer=mixer_hamiltonian)

                initial_parameters = [0] * 2 * n_layers

                best_energy_gibbs_state, _, _ = qaoa.minimize(
                    initial_p=initial_parameters,
                    method="cma",
                    options={"maxfevals": 5000},
                    initial_state=gibbs_state,
                )

                best_energy_hadamard_state, _, _ = qaoa.minimize(
                    initial_p=initial_parameters,
                    method="cma",
                    options={"maxfevals": 5000}
                )

                path_data = (
                    f"logger_data/logger_data_qaoa_performance_gibbs_states_borders_hadamard_state/logger_data_max_cut_qaoa_common_instances_n_{n_nodes}_borders/"
                    f"cmaes_qaoa_max_cut_n_{n_nodes}_t_{temperature}_"
                    f"layers_{n_layers}_case_{i}.json"
                )
                
            
                data_test = {
                    "expected_value_gibbs_state": expected_value_gibbs_state,
                    "expected_value_hadamard_state": expected_value_hadamard_state,
                    "energy_qaoa_gibbs_state": best_energy_gibbs_state,
                    "energy_qaoa_hadamard_state": best_energy_hadamard_state,
                    "energy_value_exact": exact_value,
                }


                json_datos = json.dumps(data_test, indent=2)

                with open(path_data, 'w') as archivo: 
                    archivo.write(json_datos)
                    
                i += 1

