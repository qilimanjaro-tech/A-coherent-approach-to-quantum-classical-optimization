import json
import numpy as np
import networkx as nx
import alphashape
import pandas as pd

from sklearn.decomposition import PCA
from qibo import models

from qaoa_utils import (
    exprectum_hamiltonian,
    create_gibbs_state,
    expected_value,
    build_maxcut_hamiltonian,
    build_mixer_hamiltonian,
    sample_state,
    create_gausgibbs_state,
)



def area_envolvente_concava(puntos, alpha):

    alpha_shape = alphashape.alphashape(puntos, alpha)
    
    return alpha_shape.area, alpha_shape


def compute_probabilities(init_state, alpha, beta, hamiltonian, mixer_hamiltonian):
    point_probabilities = []
    
    
    for a in alpha:
        for b in beta:
            
            final_parameters = np.array([a, b])
            
            qaoa = models.QAOA(hamiltonian=hamiltonian, mixer=mixer_hamiltonian)
            
            qaoa.set_parameters(final_parameters)
            
            quantum_state = qaoa.execute(initial_state = init_state)
            
            probabilities = np.real(quantum_state*quantum_state.conj())
            
            point_probabilities.append(probabilities)
    return point_probabilities



test = 100
test_iter = 10
n_nodes = 10
pb_conexion = 0.5
temperature = 4

alpha = np.linspace(0, np.pi, 50)
beta = np.linspace(0, 2 * np.pi, 50)

for i in range(test):
    print("case: ", i)

    delta = 0.1
    iterations = 0

    G = nx.erdos_renyi_graph(n_nodes, pb_conexion)
    
    hamiltonian = build_maxcut_hamiltonian(graph=G)
    mixer_hamiltonian = build_mixer_hamiltonian(graph=G)

    dict_values = exprectum_hamiltonian(G)
    
    gibbs_state = create_gibbs_state(dict_values, t=temperature)

    expected_value_gibbs_state = expected_value(
            state=gibbs_state, dict_values=dict_values
        )

    dict_pb_gibbs = sample_state(gibbs_state)

    pb_gibbs = list(dict_pb_gibbs.values())

    entropy_gibbs = -sum(
            pb_gibbs[i] * np.log(pb_gibbs[i])
            for i in range(len(pb_gibbs))
            if pb_gibbs[i] != 0)
    
    probs_gibbs = compute_probabilities(gibbs_state, alpha, beta, hamiltonian, mixer_hamiltonian)
    pca_gibbs = PCA(n_components = 2)
    pca_point_probabilities_gibbs = pca_gibbs.fit_transform(np.array(probs_gibbs))
    componentsDf_gibbs = pd.DataFrame(data = pca_point_probabilities_gibbs, columns = ['PC1', 'PC2'])
    area_gibbs = area_envolvente_concava(componentsDf_gibbs[['PC1','PC2']].to_numpy(), alpha = 100)[0]

    while iterations < test_iter:

        print("case: ", i, " delta: ", delta, " iterations: ", iterations)

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

        
        probs_ps_gibss = compute_probabilities(pseudo_gibbs_state, alpha, beta, hamiltonian, mixer_hamiltonian)
        pca = PCA(n_components = 2)
        pca_point_probabilities_ps_gibbs = pca.fit_transform(np.array(probs_ps_gibss))
        componentsDf_ps_gibbs = pd.DataFrame(data = pca_point_probabilities_ps_gibbs, columns = ['PC1', 'PC2'])
        area_ps_gibbs = area_envolvente_concava(componentsDf_ps_gibbs[['PC1','PC2']].to_numpy(), alpha = 100)[0]

        path_data = (f"logger_data/logger_data_qaoa_performance_gibbs_states_expressivity_same_energy/n_nodes_{n_nodes}_case_{i}_temperature_{temperature}_delta_gauss_{round(delta, 3)}.json")

        data_test = {
            "expected_value_gibbs_state": expected_value_gibbs_state,
            "expected_value_psgibbs_state": expected_value_psgibbs_state,
            "entropy_gibbs_state": entropy_gibbs,
            "entropy_psgibbs_state": entropy_psgibbs,
            "area_gibbs_state": area_gibbs,
            "area_psgibbs_state": area_ps_gibbs,
        }


        json_datos = json.dumps(data_test, indent=2)

        with open(path_data, 'w') as archivo:
            archivo.write(json_datos)

        delta += 0.1
        iterations += 1