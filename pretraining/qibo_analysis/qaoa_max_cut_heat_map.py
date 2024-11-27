import json
import numpy as np
import networkx as nx
import pandas as pd

from qaoa_utils import (
    exprectum_hamiltonian,
    create_gibbs_state,
    expected_value,
    build_maxcut_hamiltonian,
    build_mixer_hamiltonian,
    max_cut_brute_force,
    sample_state,
    create_gausgibbs_state,
    create_local_state,
    create_random_state,
    random_uniform_weights
)

from qibo import models
import qibo

qibo.set_backend("qibojit", platform="numba")

#qibo.set_threads(1)

tests = [4]
n_promedium = 2
n_nodes = 6
pb_conexion = 0.7
n_layers = 3

tempeatures = np.arange(0.1, 30.1, 0.1)
deltas = [0.4, 0.5, 0.7, 0.9, 1.5, 1.7, 2, 2.5, 3, 3.5, 4, 7, 10, 20, 30, 40]

len_deltas = len(deltas)

for i in tests:

    G = nx.erdos_renyi_graph(n_nodes, pb_conexion, seed = 0)

    exact_value = - max_cut_brute_force(G)

    hamiltonian = build_maxcut_hamiltonian(graph=G)
    mixer_hamiltonian = build_mixer_hamiltonian(graph=G)

    qaoa = models.QAOA(hamiltonian=hamiltonian, mixer=mixer_hamiltonian)

    initial_parameters = [0] * 2 * n_layers

    dict_values = exprectum_hamiltonian(G)

    hadamard_state = create_gibbs_state(dict_values, t=10**6)

    energy_hadamard_state = expected_value(state=hadamard_state, dict_values=dict_values)

    dict_pb_hadamard = sample_state(hadamard_state)

    pb_hadamard = list(dict_pb_hadamard.values())

    entropy_hadamard = -sum(
        pb_hadamard[i] * np.log(pb_hadamard[i]) for i in range(len(pb_hadamard)) if pb_hadamard[i] != 0
    )
    
    energy_qaoa_hadamard_state_list_p = []
    
    for _ in range(n_promedium):

        energy_qaoa_hadamard_state_iter, _, _ = qaoa.minimize(
            initial_p=initial_parameters,
            method="COBYLA",
            options={"maxiter": 2000},
            initial_state=hadamard_state,
        )
        
        energy_qaoa_hadamard_state_list_p.append(energy_qaoa_hadamard_state_iter)
        
    energy_qaoa_hadamard_state = sum(energy_qaoa_hadamard_state_list_p)/n_promedium
    
    one_state_final = create_local_state(solutions_dict=dict_values, reference_energy=energy_hadamard_state, left = False)

    energy_one_state_final = expected_value(state=one_state_final, dict_values=dict_values)
    
    energy_qaoa_one_state_final_list_p = []
    
    for _ in range(n_promedium):
        
        energy_qaoa_one_state_final_iter, _, _ = qaoa.minimize(
                initial_p=initial_parameters,
                method="COBYLA",
                options={"maxiter": 2000},
                initial_state=one_state_final,
            )
        
        energy_qaoa_one_state_final_list_p.append(energy_qaoa_one_state_final_iter)
    
    energy_qaoa_one_state_final = sum(energy_qaoa_one_state_final_list_p)/n_promedium
    

    for temperature in tempeatures:

        gibbs_state = create_gibbs_state(dict_values, t=temperature)

        energy_gibbs_state = expected_value(state=gibbs_state, dict_values=dict_values)

        dict_pb_gibbs = sample_state(gibbs_state)

        pb_gibbs = list(dict_pb_gibbs.values())

        entropy_gibbs = -sum(pb_gibbs[i] * np.log(pb_gibbs[i]) for i in range(len(pb_gibbs)) if pb_gibbs[i] != 0)

        energy_qaoa_gibbs_state_list_p = []
        
        for _ in range(n_promedium):
            
            energy_qaoa_gibbs_state_iter, _, _ = qaoa.minimize(
                initial_p=initial_parameters,
                method="COBYLA",
                options={"maxiter": 2000},
                initial_state=gibbs_state,)
            
            energy_qaoa_gibbs_state_list_p.append(energy_qaoa_gibbs_state_iter)
            
        energy_qaoa_gibbs_state = sum(energy_qaoa_gibbs_state_list_p)/n_promedium
        
        #--------------------------------------------------------------------------------------------------------
        
        one_state = create_local_state(solutions_dict=dict_values, reference_energy=energy_gibbs_state)

        energy_one_state = expected_value(state=one_state, dict_values=dict_values)
        
        energy_qaoa_one_state_list_p = []
        
        for _ in range(n_promedium):
        
            energy_qaoa_one_state_iter, _, _ = qaoa.minimize(
                initial_p=initial_parameters,
                method="COBYLA",
                options={"maxiter": 2000},
                initial_state=one_state,)
            
            energy_qaoa_one_state_list_p.append(energy_qaoa_one_state_iter)
            
        energy_qaoa_one_state = sum(energy_qaoa_one_state_list_p)/n_promedium
        

        list_energy_psgibbs_state = []
        list_entropy_pseudo_gibbs_state = []
        list_energy_qaoa_pseudo_gibbs_state = []
        
        list_energy_random_state = []
        list_entropy_random_state = []
        list_energy_qaoa_random_state = []

        for delta in deltas:

            print("-------------------------------------------------")
            print("Case:", i, " Temperature: ", temperature, " Delta: ", delta)
            print("-------------------------------------------------")

            pseudo_gibbs_state = create_gausgibbs_state(
                solutions_dict=dict_values,
                delta_energy=delta,
                mu_energy=energy_gibbs_state,
            )

            energy_psgibbs_state = expected_value(state=pseudo_gibbs_state, dict_values=dict_values)

            list_energy_psgibbs_state.append(energy_psgibbs_state/abs(exact_value))

            dict_pb_pseudo_gibbs_state = sample_state(pseudo_gibbs_state)

            pb_pseudo_gibbs_state = list(dict_pb_pseudo_gibbs_state.values())

            entropy_pseudo_gibbs_state = -sum(
                pb_pseudo_gibbs_state[i] * np.log(pb_pseudo_gibbs_state[i])
                for i in range(len(pb_pseudo_gibbs_state))
                if pb_pseudo_gibbs_state[i] != 0
            )

            list_entropy_pseudo_gibbs_state.append(entropy_pseudo_gibbs_state/entropy_hadamard)
            
            energy_qaoa_pseudo_gibbs_state_list_p = []
            
            for _ in range(n_promedium):

                energy_qaoa_pseudo_gibbs_state_iter, _, _ = qaoa.minimize(
                    initial_p=initial_parameters,
                    method="COBYLA",
                    options={"maxiter": 2000},
                    initial_state=pseudo_gibbs_state,)
                
                energy_qaoa_pseudo_gibbs_state_list_p.append(energy_qaoa_pseudo_gibbs_state_iter)
                
            energy_qaoa_pseudo_gibbs_state = sum(energy_qaoa_pseudo_gibbs_state_list_p)/n_promedium

            list_energy_qaoa_pseudo_gibbs_state.append(energy_qaoa_pseudo_gibbs_state/abs(exact_value))

            #--------------------------------------------------------------------------------------------------------

                
            random_state = create_random_state(solutions_dict = dict_values, max_energy = energy_gibbs_state)
            
            energy_random_state = expected_value(state=random_state, dict_values=dict_values)
            
            list_energy_random_state.append(energy_random_state/abs(exact_value))
            
            dict_pb_random = sample_state(random_state)

            pb_random = list(dict_pb_random.values())

            entropy_random_state = -sum(pb_random[i] * np.log(pb_random[i]) for i in range(len(pb_random)) if pb_random[i] != 0)
            
            list_entropy_random_state.append(entropy_random_state/entropy_hadamard)
            
            energy_qaoa_random_state_list_p = []
            
            for _ in range(n_promedium):
            
                energy_qaoa_random_state_iter, _, _ = qaoa.minimize(
                        initial_p=initial_parameters,
                        method="COBYLA",
                        options={"maxiter": 2000},
                        initial_state=random_state,)
                
                energy_qaoa_random_state_list_p.append(energy_qaoa_random_state_iter)
                
            energy_qaoa_random_state = sum(energy_qaoa_random_state_list_p)/n_promedium
            
            list_energy_qaoa_random_state.append(energy_qaoa_random_state/abs(exact_value))
            #--------------------------------------------------------------------------------------------------------
        
        
        data_frame = {
            "case": [i] * len_deltas,
            "exact_energy": [exact_value] * len_deltas,
            "energy_hadamard_state": [energy_hadamard_state/abs(exact_value)] * len_deltas,
            "entropy_hadamard_state": [entropy_hadamard/entropy_hadamard] * len_deltas,
            "energy_qaoa_hadamard_state": [energy_qaoa_hadamard_state/abs(exact_value)] * len_deltas,
            "energy_one_state_final": [energy_one_state_final/abs(exact_value)] * len_deltas,
            "entropy_one_state_final": [0] * len_deltas,
            "energy_qaoa_one_state_final": [energy_qaoa_one_state_final/abs(exact_value)] * len_deltas,
            "temperature_gibbs_state": [temperature] * len_deltas,
            "energy_gibbs_state": [energy_gibbs_state/abs(exact_value)] * len_deltas,
            "entropy_gibbs_state": [entropy_gibbs/entropy_hadamard] * len_deltas,
            "energy_qaoa_gibbs_state": [energy_qaoa_gibbs_state/abs(exact_value)] * len_deltas,
            "energy_one_state": [energy_one_state/abs(exact_value)] * len_deltas,
            "entropy_one_state": [0] * len_deltas,
            "energy_qaoa_one_state": [energy_qaoa_one_state/abs(exact_value)] * len_deltas,
            "energy_random_state": list_energy_random_state,
            "entropy_random_state": list_entropy_random_state,
            "energy_qaoa_random_state": list_energy_qaoa_random_state,
            "energy_pseudo_gibbs_state": list_energy_psgibbs_state,
            "entropy_pseudo_gibbs_state": list_entropy_pseudo_gibbs_state,
            "energy_qaoa_pseudo_gibbs_state": list_energy_qaoa_pseudo_gibbs_state,
        }

        df = pd.DataFrame(data_frame)

        path = f"logger_data/logger_data_qaoa_performance_gibbs_states_heat_map/case_{i}_heat_map_n_{n_nodes}_{n_layers}_layers_temperature_{round(temperature, 3)}.json"

        df.to_json(path, orient="records")
    