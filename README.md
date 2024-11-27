# A-coherent-approach-to-quantum-classical-optimization

This repository contains the code used to produce the results of the publication [A coherent approach to quantum-classical optimization](https://arxiv.org/abs/2409.13924).

## Results and data verification

- The data used to create the figures in the article is located in the `pretraining/logger_data` folder.

- Within the `logger_data` folder, there are several subfolders, each containing the data necessary to reproduce the results presented in the article. See the associated readme for more information.

- The code used to generate the figures can be found in the file `plots_from_data.ipynb`.

- In the file `classical_quantum_optimisation.ipynb` we present functional examples of hybrid optimization schemes. We also added two files presenting the operation of the DMRG algorithm and the WI,II MPO called `dmrg.ipynb` and `mpo_time_evolution.ipynb` respectively.


## Code 

The code used is divided into two main modules. 

- The first module is located within the `qibo_analysis` folder. This folder contains the code necessary to conduct the study on the performance of pure Gibbs states as initialization states. 

- The second module is located within the `variational_algorithms` folder. This folder contains the code that implements the combined optimization protocol for tensor networks and VQA. It includes both the new protocol introduced and the state-of-the-art protocol used for comparison.


## Installation

To correctly install the dependencies, please create an environment with python = 3.10.


```ruby
conda create -n pretraining_env python=3.10
conda activate pretraining_env
```

And install the corresponding packages

```ruby
pip install -r requirements.txt
```

## Citation

```ruby
@misc{cáliz2024coherentapproachquantumclassicaloptimization,
      title={A coherent approach to quantum-classical optimization}, 
      author={Andrés N. Cáliz and Jordi Riu and Josep Bosch and Pau Torrente and Jose Miralles and Arnau Riera},
      year={2024},
      eprint={2409.13924},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2409.13924}, 
}
```