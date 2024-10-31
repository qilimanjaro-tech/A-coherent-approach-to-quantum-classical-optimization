import json
import networkx as nx
import qibo
from qaoa_utils import (
    exprectum_hamiltonian_brute_force,
    create_gibbs_state,
    expected_value,
    build_maxcut_hamiltonian,
    build_mixer_hamiltonian,
    max_cut_brute_force,
    create_local_state,
)
from qibo import models

qibo.set_threads(1)

test = 30
pb_conexion = 0.65
node_list = [8, 10, 12, 14, 16]
temperature_list = [6]

n_layers_list = [3]

for temperature in temperature_list:

    for z in range(len(n_layers_list)):
            
        n_layers = n_layers_list[z]

        for n_nodes in node_list:
            
            i = 0
            
            while i < test:
                
                print('-----------------------------------------')
                print("case: ", i,)
                print('-----------------------------------------')

                G = nx.erdos_renyi_graph(n_nodes, pb_conexion)

                dict_values = exprectum_hamiltonian_brute_force(G)

                gibbs_state = create_gibbs_state(dict_values, t=temperature)

                expected_value_gibbs_state = expected_value(
                    state=gibbs_state, dict_values=dict_values
                )


                one_state = create_local_state(
                    solutions_dict=dict_values, reference_energy=expected_value_gibbs_state
                )

                expected_value_one_state = expected_value(state=one_state, dict_values=dict_values)

                exact_value = - max_cut_brute_force(G)

                if exact_value != expected_value_one_state:

                    hamiltonian = build_maxcut_hamiltonian(graph=G)
                    mixer_hamiltonian = build_mixer_hamiltonian(graph=G)

                    qaoa = models.QAOA(hamiltonian=hamiltonian, mixer=mixer_hamiltonian)

                    initial_parameters = [0] * 2 * n_layers

                    best_energy_gibbs_state, _, _ = qaoa.minimize(
                        initial_p=initial_parameters,
                        method="COBYLA",
                        options={"maxiter": 8000},
                        initial_state=gibbs_state,
                    )

                    best_energy_one_state, _, _ = qaoa.minimize(
                        initial_p=initial_parameters,
                        method="COBYLA",
                        options={"maxiter": 8000},
                        initial_state=one_state,
                    )

                    path_data = (
                        f"logger_data/logger_data_qaoa_performance_gibbs_states_borders_one_state/logger_data_max_cut_qaoa_common_instances_n_{n_nodes}_borders/"
                        f"cmaes_qaoa_max_cut_n_{n_nodes}_t_{temperature}_"
                        f"layers_{n_layers}_case_{i}.json"
                    )
                    
                
                    data_test = {
                        "expected_value_gibbs_state": expected_value_gibbs_state,
                        "expected_value_one_state": expected_value_one_state,
                        "energy_qaoa_gibbs_state": best_energy_gibbs_state,
                        "energy_qaoa_one_state": best_energy_one_state,
                        "energy_value_exact": exact_value,
                    }


                    json_datos = json.dumps(data_test, indent=2)

                    with open(path_data, 'w') as archivo: 
                        archivo.write(json_datos)
                        
                    i += 1

