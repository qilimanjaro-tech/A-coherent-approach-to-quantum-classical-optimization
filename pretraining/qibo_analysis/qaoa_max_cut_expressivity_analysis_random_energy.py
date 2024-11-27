import numpy as np
import networkx as nx
import pandas as pd


from qibo import models
from qaoa_utils import exprectum_hamiltonian, sample_state, build_maxcut_hamiltonian, build_mixer_hamiltonian, create_random_state_no_energy
from sklearn.decomposition import PCA
import alphashape


def area_envolvente_concava(puntos, alpha):

    alpha_shape = alphashape.alphashape(puntos, alpha)
    return alpha_shape.area, alpha_shape


def compute_probabilities(init_state, alpha, beta):
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

n = 8
p = 0.5


alpha = np.linspace(0, np.pi, 50)
beta = np.linspace(0, 2 * np.pi, 50)
areas=[]

for j in range(100):
    
    print('case', j)
    
    flag = 1
    while flag > 0:
        G = nx.erdos_renyi_graph(n, p)
        flag = sum([degree==0 for node,degree in G.degree])
        
    hamiltonian = build_maxcut_hamiltonian(graph = G)
    mixer_hamiltonian = build_mixer_hamiltonian(graph = G)
    dict_values = exprectum_hamiltonian(G)
    
    for i in range(10):
        
        print('case', j, 'number', i)
        
        state = create_random_state_no_energy(solutions_dict = dict_values)
        dict_pb_psgibbs = sample_state(state)

        pb_psgibbs = list(dict_pb_psgibbs.values())

        entropy = -sum(
            pb_psgibbs[i] * np.log2(pb_psgibbs[i])
            for i in range(len(pb_psgibbs))
            if pb_psgibbs[i] != 0
                )
        probs = compute_probabilities(state, alpha, beta)
        pca = PCA()
        try:
            pca.fit(np.array(probs))
        except:
            continue
        pca_point_probabilities = pca.transform(np.array(probs))[:,:2]
        svalues = pca.singular_values_
        componentsDf = pd.DataFrame(data = pca_point_probabilities, columns = ['PC1', 'PC2'])
        gamma=100
        rank = (svalues>1e-2).sum()
        area = area_envolvente_concava(componentsDf[['PC1','PC2']].to_numpy(), gamma)[0]
        areas.append([j, i, entropy,area,rank])
        print(j,i, entropy, area, rank)

areas_entropy_df = pd.DataFrame(areas, columns=['Instance', 'State', 'Entropy', 'Area', 'Rank'])
    #pintar_envolvente_concava(componentsDf[['PC1','PC2']].to_numpy(), gamma)
areas_entropy_df.to_csv('logger_data/Results',index=False)