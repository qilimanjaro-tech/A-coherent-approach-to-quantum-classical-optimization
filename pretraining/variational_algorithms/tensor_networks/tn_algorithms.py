from typing import Optional, Union, List, Tuple, Literal
from types import FunctionType
from copy import deepcopy
from collections import defaultdict
import itertools
import numpy as np
from numpy import complex128
from numpy import linalg as LA
import scipy.linalg as la
import random
from scipy.linalg import null_space
from ncon import ncon

from variational_algorithms.hamiltonians import Hamiltonian
from variational_algorithms.tensor_networks.tensor_networks import TensorOperations, MPS, MPO, MPSGenerator, TensorGates
from variational_algorithms.utils import KAKDecomposition
from variational_algorithms.tensor_networks.tn_utils import StructureResultsPretraining

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import library as gates


class DMRG:
    """
    Class representing the Density Matrix Renormalization Group (DMRG) algorithm.

    Args:
        - mpo (MPO): The Matrix Product Operator (MPO) representing the Hamiltonian.
        - chi (int): The maximum bond dimension of the MPS tensors.
        - sweeps (int): The number of sweeps to perform in the DMRG algorithm.
        - lanczos_maxit (int, optional): The maximum number of Lanczos iterations. Defaults to 2.
        - lanczos_krydim (int, optional): The dimension of the Krylov subspace in the Lanczos method. Defaults to 4.

    Attributes:
        - left_mps (MPS): The left orthogonal form of the MPS.
        - right_mps (MPS): The right orthogonal form of the MPS.
        - mpo (MPO): The Matrix Product Operator (MPO) representing the Hamiltonian.
        - L (np.ndarray): Array of tensors representing the left environment blocks of each site.
        - R (np.ndarray): Array of tensors representing the right environment blocks of each site.
        - chi (int): The maximum bond dimension of the MPS tensors.
        - n_sites (int): The number of sites in the MPS.
        - d (int): The local Hilbert space dimension.
        - sweeps (int): The number of sweeps to perform in the DMRG algorithm.
        - sWeight (list): List of Schmidt coefficients at each bond.
        - krydim (int): The dimension of the Krylov subspace in the Lanczos method.
        - maxit (int): The maximum number of Lanczos iterations.
        - energy (list): List of energies at each update step.

    Methods:
        - twosite_applympo(psi: np.ndarray, p: int, rank1: bool = True) -> np.ndarray:
            Apply the Hamiltonian to the MPS state using a 2-site update strategy.

        - left_right_2site_update(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
            Update the MPS tensors at position p and p+1 on a left to right sweep using the Lanczos method.

        - right_left_2site_update(p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
            Update the MPS tensors at position p and p+1 on a right to left sweep using the Lanczos method.

        - leftright_sweep():
            Perform a full left to right sweep of the MPS, updating each 2-site tensor using the Lanczos method.

        - rightleft_sweep():
            Perform a full right to left sweep of the MPS, updating each 2-site tensor using the Lanczos method.

        - run() -> Tuple[MPS, MPS, List[np.ndarray], list]:
            Run the full DMRG algorithm by sweeping the MPS left to right and right to left, updating the tensors using
            the Lanczos method.

    """

    def __init__(
        self,
        mpo: MPO,
        chi: int,
        sweeps: int,
        lanczos_maxit: int = 2,
        lanczos_krydim: int = 4,
        metrics: bool = False,
        prep_state: Union[
            str("Decay_state"), str("Uniform_state"), str("Two_state"), str("One_state"), str("Gibbs_state") # type: ignore
        ] = "One_state",
    ):  # type: ignore

        self.left_mps = MPS(mpo.n_sites, mpo.d, chi)
        self.left_mps.ones_init()
        self.left_mps.normalize()
        self.left_mps.left_orthogonalize()

        self.right_mps = MPS(mpo.n_sites, mpo.d, chi)
        self.right_mps.ones_init()
        self.right_mps.normalize()
        self.right_mps.right_orthogonalize()

        self.mpo = mpo
        self.l = np.array([None for _ in range(self.mpo.n_sites)])
        self.r = np.array([None for _ in range(self.mpo.n_sites)])

        self.chi = chi
        self.n_sites = mpo.n_sites
        self.d = mpo.d
        self.sweeps = sweeps
        self.sweight = [0 for _ in range(self.n_sites - 1)]
        self.krydim = lanczos_krydim
        self.maxit = lanczos_maxit

        self.energy = []

        self.prep_state = prep_state
        self.metrics = metrics
        self.coherence = []
        self.effective_dimension = []
        self.list_mps_history = []

    def twosite_applympo(self, psi: np.ndarray, p: int, rank1: bool = True) -> np.ndarray:
        """
        Apply the hamiltonian to the mps state, using a 2-site update strategy. psi encodes site p and p+1 of the mps
        reshaped into a rank-1 tensor and the rest of the mpo and mps are encoded in the L and R tensors.

        Args:
            - psi (np.ndarray): rank-1 tensor encoding site p and p+1 of the mps.
            - p (int): position of the first mps tensor in the whole mps.
            - rank1 (bool, optional): whether to return the updated psi as a rank-1 tensor again or not.
                Defaults to True.

        Returns:
            - np.ndarray: updated 2-site mps tensor psi.

        """

        if p == 0:
            psi_reshaped = psi.reshape(1, self.mpo[p].shape[0], self.mpo[p + 1].shape[1], self.r[p + 2].shape[0])

            psi_mpo = ncon(
                [psi_reshaped, self.mpo[p], self.mpo[p + 1], self.r[p + 2]],
                [[-1, 1, 3, 4], [1, -2, 2], [2, 3, -3, 5], [4, 5, -4]],
            )

            return (
                psi_mpo.reshape(self.mpo[p].shape[0] * self.mpo[p + 1].shape[1] * self.r[p + 2].shape[0])
                if rank1
                else psi_mpo
            )

        elif p == self.n_sites - 2:
            psi_reshaped = psi.reshape(self.l[p - 1].shape[0], self.mpo[p].shape[1], self.mpo[p + 1].shape[1], 1)

            psi_mpo = ncon(
                [self.l[p - 1], psi_reshaped, self.mpo[p], self.mpo[p + 1]],
                [[1, 2, -1], [1, 3, 5, -4], [2, 3, -2, 4], [4, 5, -3]],
            )

            return (
                psi_mpo.reshape(self.l[p - 1].shape[0] * self.mpo[p].shape[1] * self.mpo[p + 1].shape[1])
                if rank1
                else psi_mpo
            )

        else:
            psi_reshaped = psi.reshape(
                self.l[p - 1].shape[0], self.mpo[p].shape[1], self.mpo[p + 1].shape[1], self.r[p + 2].shape[0]
            )

            psi_mpo = ncon(
                [self.l[p - 1], psi_reshaped, self.mpo[p], self.mpo[p + 1], self.r[p + 2]],
                [[1, 2, -1], [1, 3, 5, 6], [2, 3, -2, 4], [4, 5, -3, 7], [6, 7, -4]],
            )
            return (
                psi_mpo.reshape(
                    self.l[p - 1].shape[0] * self.mpo[p].shape[1] * self.mpo[p + 1].shape[1] * self.r[p + 2].shape[0]
                )
                if rank1
                else psi_mpo
            )

    def left_right_2site_update(self, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update the mps tensors at position p and p+1 on a left to right sweep using Lanczos method to obtain the
        2-site ground state.

        Args:
            - p (int): position of the first mps tensor to be updated in the whole mps

        Returns:
            - (np.ndarray, np.ndarray, np.ndarray, int, int): 3 tensors from the SVD of the updated 2-site mps tensor.

        """

        chi_l = self.right_mps[p].shape[0]
        chi_r = self.right_mps[p + 1].shape[2]

        if p == 0:
            psiground = ncon([self.right_mps[p], self.right_mps[p + 1]], [[-1, -2, 1], [1, -3, -4]]).reshape(
                (chi_l * chi_r * self.d * self.d,),
            )

        else:
            assert self.sweight[p - 1].shape[1] == self.right_mps[p].shape[0]

            psiground = ncon(
                [self.sweight[p - 1], self.right_mps[p], self.right_mps[p + 1]], [[-1, 1], [1, -2, 2], [2, -3, -4]]
            ).reshape(
                (chi_l * chi_r * self.d * self.d,),
            )

        psiground, entemp = self._eig_lanczos(
            psiground,
            self.twosite_applympo,
            (p, True),
            self.maxit,
            self.krydim,
        )

        self.energy.append(entemp + self.mpo.offset)

        utemp, stemp, vhtemp = la.svd(psiground.reshape(chi_l * self.d, self.d * chi_r), full_matrices=False)

        return utemp, stemp, vhtemp

    def right_left_2site_update(self, p: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update the mps tensors at position p and p+1 on a right to left sweep using Lanczos method to obtain the
        2-site ground state.

        Args:
            - p (int): position of the first mps tensor to be updated in the whole mps

        Returns:
            - (np.ndarray, np.ndarray, np.ndarray): 3 tensors from the SVD of the updated 2-site mps tensor.
        """

        chi_l = self.left_mps[p].shape[0]
        chi_r = self.left_mps[p + 1].shape[2]

        if p == self.n_sites - 2:
            psiground = ncon([self.left_mps[p], self.left_mps[p + 1]], [[-1, -2, 1], [1, -3, -4]]).reshape(
                (chi_l * chi_r * self.d * self.d,),
            )
        else:
            assert self.left_mps[p + 1].shape[2] == self.sweight[p + 1].shape[0]

            psiground = ncon(
                [self.left_mps[p], self.left_mps[p + 1], self.sweight[p + 1]], [[-1, -2, 1], [1, -3, 2], [2, -4]]
            ).reshape(
                (chi_l * chi_r * self.d * self.d,),
            )

        psiground, entemp = self._eig_lanczos(
            psiground,
            self.twosite_applympo,
            (p, True),
            self.maxit,
            self.krydim,
        )
        self.energy.append(entemp + self.mpo.offset)
        utemp, stemp, vhtemp = la.svd(psiground.reshape(chi_l * self.d, self.d * chi_r), full_matrices=False)

        return utemp, stemp, vhtemp

    def leftright_sweep(self):
        """
        Complete a full left to right sweep of the mps, updating each 2-site tensor using Lanczos method.

        """

        for site in range(self.n_sites - 1):
            chi_l = self.right_mps[site].shape[0]
            chi_r = self.right_mps[site + 1].shape[2]

            utemp, stemp, vhtemp = self.left_right_2site_update(site)

            # Truncate the bond dimension if necessary
            chitemp = min(len(stemp), self.chi)

            # Update the left tensor
            self.left_mps[site] = utemp[:, range(chitemp)].reshape(chi_l, self.d, chitemp)

            # Truncate and normalize the Schmidt coefficients
            self.sweight[site] = np.diag(stemp[range(chitemp)] / la.norm(stemp[range(chitemp)]))

            # Update the right tensor using the right tensor from the SVD and the Schmidt coefficients
            self.right_mps[site + 1] = vhtemp[range(chitemp), :].reshape(chitemp, self.d, chi_r)

            # Update the left environment block using utemp, taking into account if it is the first site or not
            if site == 0:
                self.l[site] = ncon(
                    [self.left_mps[site], self.mpo[site], np.conj(self.left_mps[site])],
                    [[1, 2, -1], [2, 3, -2], [1, 3, -3]],
                )
            else:
                self.l[site] = ncon(
                    [self.l[site - 1], self.left_mps[site], self.mpo[site], np.conj(self.left_mps[site])],
                    [[1, 2, 3], [1, 4, -1], [2, 4, 5, -2], [3, 5, -3]],
                )

        self.left_mps[self.n_sites - 1] = ncon(
            [self.sweight[self.n_sites - 2], self.right_mps[self.n_sites - 1]], [[-1, 1], [1, -2, -3]]
        )

    def rightleft_sweep(self):
        """
        Complete a full right to left sweep of the mps, updating each 2-site tensor using Lanczos method.

        """

        for site in range(self.n_sites - 1, 0, -1):
            chi_l = self.left_mps[site - 1].shape[0]
            chi_r = self.left_mps[site].shape[2]

            utemp, stemp, vhtemp = self.right_left_2site_update(site - 1)

            # Truncate the bond dimension if necessary
            chitemp = min(len(stemp), self.chi)

            # Update the right tensor
            self.right_mps[site] = vhtemp[range(chitemp), :].reshape(chitemp, self.d, chi_r)

            # Truncate and normalize the Schmidt coefficients
            self.sweight[site - 1] = np.diag(stemp[range(chitemp)] / la.norm(stemp[range(chitemp)]))
            self.left_mps[site - 1] = utemp[:, range(chitemp)].reshape(chi_l, self.d, chitemp)

            # Update the right environment block using vhtemp, taking into account if it is the last site or not
            if site == self.n_sites - 1:
                self.r[site] = ncon(
                    [self.right_mps[site], self.mpo[site], np.conj(self.right_mps[site])],
                    [[-1, 1, 2], [-2, 1, 3], [-3, 3, 2]],
                )
            else:
                self.r[site] = ncon(
                    [self.right_mps[site], self.mpo[site], self.r[site + 1], np.conj(self.right_mps[site])],
                    [[-1, 1, 2], [-2, 1, 3, 4], [2, 4, 5], [-3, 3, 5]],
                )

        self.right_mps[0] = ncon([self.left_mps[0], self.sweight[0]], [[-1, -2, 1], [1, -3]])

    def run(self) -> Tuple[MPS, MPS, List[np.ndarray], list]:
        """
        Run the full DMRG algorithm by sewwping the mps left to right and right to left, updating the tensors using
        Lanczos methods with two tensors at a time.

        Returns:
            - (MPS, MPS, List[np.ndarray], list): MPS encoding the ground state of the given Hamiltonian
                in MPO form in left and right orthogonal form, aswell as the Schmidt coefficient
                from the SVD at each bond and the energy at each update step.

        """

        self.r[-1] = ncon(
            [self.right_mps[-1], self.mpo[-1], np.conj(self.right_mps[-1])], [[-1, 1, 2], [-2, 1, 3], [-3, 3, 2]]
        )

        for s in range(self.n_sites - 2, 1, -1):
            self.r[s] = ncon(
                [self.right_mps[s], self.mpo[s], self.r[s + 1], np.conj(self.right_mps[s])],
                [[-1, 1, 2], [-2, 1, 3, 4], [2, 4, 5], [-3, 3, 5]],
            )

        for _ in range(self.sweeps):
            self.leftright_sweep()
            self.rightleft_sweep()

        self.left_mps = deepcopy(self.right_mps)
        self.left_mps.left_orthogonalize()

        if not self.metrics:

            self.coherence = None
            self.effective_dimension = None

        else:

            dict_sample_right = SamplingMPS().sampling(mps=self.left_mps.tensors, n_samples=2000)

            pb_left = list(dict_sample_right.values())

            entropy = -sum(pb_left[i] * np.log2(pb_left[i]) for i in range(len(pb_left)))

            effective_dimension_value = (sum(pb_left[i] ** 2 for i in range(len(pb_left)))) ** (-1)

            self.list_mps_history.append(self.right_mps.tensors)

            self.coherence.append(entropy)
            self.effective_dimension.append(effective_dimension_value)

        return (self.left_mps, self.right_mps, self.sweight, self.energy, self.coherence, self.effective_dimension)

    def _eig_lanczos(
        self,
        input_psivec: np.ndarray,
        linfunc: FunctionType,
        funcargs: Optional[tuple],
        maxit: int = 2,
        krydim: int = 4,
    ):
        """
        Compute the eigenvalues and eigenvectors of a linear operator using the Lanczos algorithm.
        Args:
            - input_psivec (np.ndarray): The initial vector for the Lanczos algorithm.
            - linfunc (FunctionType): The linear operator function. For the case of DMRG, this is the
                function that applies the MPO to the MPS.
            - funcargs (Optional[tuple]): Additional arguments to be passed to the linear operator function.
                In the case of DMRG, this is the tuple containing the position in the MPS and whether to
                return the updated tensor as a rank-1 tensor.
            - maxit (int): The maximum number of Lanczos iterations. Defaults to 2.
            - krydim (int): The dimension of the Krylov subspace. Defaults to 4.
        Returns:
            (np.ndarray, float): A tuple containing the eigenvector and eigenvalue.
        """
        psivec = input_psivec.copy()
        if la.norm(psivec) == 0:
            psivec = np.random.rand(len(psivec))
        psi = np.zeros([len(psivec), krydim + 1])
        a = np.zeros([krydim, krydim])
        dval = 0
        for _ in range(maxit):
            psi[:, 0] = psivec / max(la.norm(psivec), 1e-16)
            for ip in range(1, krydim + 1):
                psi[:, ip] = linfunc(psi[:, ip - 1], *funcargs)
                for ig in range(ip):
                    a[ip - 1, ig] = np.dot(psi[:, ip], psi[:, ig])
                    a[ig, ip - 1] = np.conj(a[ip - 1, ig])
                for ig in range(ip):
                    psi[:, ip] = psi[:, ip] - np.dot(psi[:, ig], psi[:, ip]) * psi[:, ig]
                    psi[:, ip] = psi[:, ip] / max(la.norm(psi[:, ip]), 1e-16)

            [dtemp, utemp] = la.eigh(a)

            if self.prep_state == "One_state":

                psivec = psi[:, range(0, krydim)] @ utemp[:, 0]

            elif self.prep_state == "Gibbs_state":

                psivec = np.exp(-dtemp[0]) * psi[:, range(0, krydim)] @ utemp[:, 0]

                for i in range(1, len(dtemp)):
                    psivec += np.exp(-dtemp[i]) * psi[:, range(0, krydim)] @ utemp[:, i]

            elif self.prep_state == "Two_state":

                psivec = psi[:, range(0, krydim)] @ utemp[:, 0]
                psivec += psi[:, range(0, krydim)] @ utemp[:, 1]

            elif self.prep_state == "Decay_state":

                psivec = psi[:, range(0, krydim)] @ utemp[:, 0]

                for i in range(1, len(dtemp)):
                    psivec += 1 / (i + 1) * psi[:, range(0, krydim)] @ utemp[:, i]

            elif self.prep_state == "Uniform_state":

                psivec = psi[:, range(0, krydim)] @ utemp[:, 0]

                for i in range(1, len(dtemp)):
                    psivec += psi[:, range(0, krydim)] @ utemp[:, i]

        psivec = psivec / la.norm(psivec)
        dval = dtemp[0]

        return psivec, dval


class TimeEvolution:
    """
    Class representing the MPO Time Evolution algorithm.
    """
    def __init__(
        self,
        n_sites: int,
        ham: Hamiltonian,
        chi: int,
        init_mps: MPS = None,
        stop_energy: float = None,
    ):
        # TODO: Check open_bounds args for MPO and MPS classes
        self.n_sites = n_sites
        
        self.ham = ham
        self.ham_mpo = MPO(n_sites=self.n_sites, phys_d=2, chi=chi, open_bounds=False)
        self.ham_mpo.from_ham(ham * (-1))
        
        self.chi = chi
        
        self.stop_energy = stop_energy

        if init_mps is not None:
            self.init_mps = init_mps
        else: # If no MPS is specified, the initial state is set to the Hadamard state.
            self.init_mps = MPS(self.n_sites, 2, self.chi) # Check if open
            tensors = MPSGenerator(d = 2, chi = chi, n_sites = self.n_sites).zeros_mps()
            for i in range(self.n_sites):
                tensors[i] = ncon([tensors[i], TensorGates().h()], [[-1, 1, -3], [1, -2]])
            self.init_mps.tensors = tensors

        self.energies = []

    def run(self, dt: float, n_steps: int, order: int=1, metrics=True):
        
        self.dt = dt
        self.n_steps = n_steps

        if order not in [1, 2]:
            raise ValueError("Only approximations of order 1 or 2 are supported")
        elif order == 1:
            U_tensors = self.get_WI_list()
        elif order == 2:
            U_tensors = self.get_WII_list()

        energies = []
        
        self.coherence = []
        self.effective_dimension = []
        self.list_mps_history = []
        
        new_mps = deepcopy(self.init_mps.tensors)
        
        for _ in range(self.n_steps):
            
            new_mps[0] = ncon([new_mps[0], U_tensors[0]], [[-1, 1, -3], [1, -2, -4]])
            new_mps[-1] = ncon([new_mps[-1], U_tensors[-1]], [[-1, 1, -4], [-2, 1, -3]])

            for i in range(1, self.n_sites-1):
                new_mps[i] = ncon([new_mps[i], U_tensors[i]], [[-1, 1, -4], [-2, 1, -3, -5]])

            shape = (new_mps[0].shape[0], new_mps[0].shape[1], new_mps[0].shape[2]*new_mps[0].shape[3])
            new_mps[0] = new_mps[0].reshape(shape)

            shape = (new_mps[-1].shape[0]*new_mps[-1].shape[1], new_mps[-1].shape[2], new_mps[-1].shape[3])
            new_mps[-1] = new_mps[-1].reshape(shape)
            for i in range(1, self.n_sites-1):
                shape = (new_mps[i].shape[0]*new_mps[i].shape[1], new_mps[i].shape[2],
                         new_mps[i].shape[3]*new_mps[i].shape[4])
                new_mps[i] = new_mps[i].reshape(shape)
            
            new_mps = TensorOperations(new_mps).truncation_network_left_form(chi=self.chi)

            normalized_mps = MPS(self.n_sites, 2, self.chi)
            normalized_mps.tensors = new_mps
            normalized_mps.normalize()
            
            energies.append(-(normalized_mps.mpo_expval(self.ham_mpo)).real)
            
            new_mps = normalized_mps.tensors
            
            self.list_mps_history.append(new_mps)
            
            if self.stop_energy!=None and energies[-1] < self.stop_energy:
                
                if not metrics:

                    self.coherence = None
                    self.effective_dimension = None

                else:

                    dict_sample = SamplingMPS().sampling(mps=new_mps, n_samples=2000)

                    pb = list(dict_sample.values())

                    entropy = -sum(pb[i] * np.log2(pb[i]) for i in range(len(pb)))

                    effective_dimension_value = (sum(pb[i] ** 2 for i in range(len(pb)))) ** (-1)

                    self.coherence.append(entropy)
                    self.effective_dimension.append(effective_dimension_value)
                
                self.energies = energies

                return normalized_mps

        if not metrics:

            self.coherence = None
            self.effective_dimension = None

        else:

            dict_sample = SamplingMPS().sampling(mps=new_mps, n_samples=2000)

            pb = list(dict_sample.values())

            entropy = -sum(pb[i] * np.log2(pb[i]) for i in range(len(pb)))

            effective_dimension_value = (sum(pb[i] ** 2 for i in range(len(pb)))) ** (-1)

            self.coherence.append(entropy)
            self.effective_dimension.append(effective_dimension_value)
        
        self.energies = energies

        return normalized_mps

    def get_WI_list(self):
        W_list = deepcopy(self.ham_mpo.tensors)
        W_list[0] = W_list[0].transpose(2,0,1)
        for i in range(1, len(W_list)-1):
            W_list[i] = W_list[i].transpose(0,3,1,2)

        WI_list = []

        D = W_list[0][-1]
        C = W_list[0][1:-1]

        site_tensor = np.zeros((C.shape[0]+1, 2, 2), dtype=complex)
        site_tensor[0] = np.eye(2) + self.dt * D
        site_tensor[1:] = np.sqrt(self.dt) * C

        WI_list.append(site_tensor)

        for i in range(1, self.n_sites-1):
            D = W_list[i][0, -1]
            C = W_list[i][0, 1:-1]
            B = W_list[i][1:-1, -1]
            A = W_list[i][1:-1, 1:-1]

            site_tensor = np.zeros((A.shape[0]+1, A.shape[1]+1, 2, 2), dtype=complex)
            site_tensor[0, 0] = np.eye(2) + self.dt * D
            site_tensor[0, 1:] = np.sqrt(self.dt) * C
            site_tensor[1:, 0] = np.sqrt(self.dt) * B
            site_tensor[1:, 1:] = A

            WI_list.append(site_tensor)

        D = W_list[-1][0]
        B = W_list[-1][1:-1]

        site_tensor = np.zeros((B.shape[0]+1, 2, 2), dtype=complex)
        site_tensor[0] = np.eye(2) + self.dt * D
        site_tensor[1:] = np.sqrt(self.dt) * B

        WI_list.append(site_tensor)

        WI_list[0] = WI_list[0].transpose(1,2,0)
        for i in range(1, self.n_sites-1):
            WI_list[i] = WI_list[i].transpose(0,2,3,1)

        return WI_list

    def get_WII_list(self):
        W_list = deepcopy(self.ham_mpo.tensors)
        W_list[0] = W_list[0].transpose(2,0,1)
        for i in range(1, len(W_list)-1):
            W_list[i] = W_list[i].transpose(0,3,1,2)

        WII_list = []

        D = W_list[0][-1]
        C = W_list[0][1:-1]

        WD = la.expm(self.dt * D)
        WC = np.zeros(C.shape, dtype=complex)

        for i in range(WC.shape[0]):
            F = np.zeros((4, 2, 4, 2), dtype=complex)

            for k in range(4):
                F[k, :, k, :] = self.dt * D
            F[1, :, 0, :] = np.sqrt(self.dt) * C[i]
            F[3, :, 2, :] = np.sqrt(self.dt) * C[i]

            Fexp = la.expm(F.reshape((8,8))).reshape((4, 2, 4, 2))[:, :, 0, :]

            WC[i] = Fexp[1]

        site_tensor = np.zeros((C.shape[0]+1, 2, 2), dtype=complex)
        site_tensor[0] = WD
        site_tensor[1:] = WC

        WII_list.append(site_tensor)

        for site in range(1, self.n_sites-1):
            D = W_list[site][0, -1]
            C = W_list[site][0, 1:-1]
            B = W_list[site][1:-1, -1]
            A = W_list[site][1:-1, 1:-1]

            WA = np.zeros(A.shape, dtype=complex)
            WB = np.zeros(B.shape, dtype=complex)
            WC = np.zeros(C.shape, dtype=complex)
            WD = la.expm(self.dt * D)

            for i in range(WA.shape[0]):
                for j in range(WA.shape[1]):
                    F = np.zeros((4, 2, 4, 2), dtype=complex)

                    for k in range(4):
                        F[k, :, k, :] = self.dt * D
                    F[1, :, 0, :] = np.sqrt(self.dt) * C[j]
                    F[3, :, 2, :] = np.sqrt(self.dt) * C[j]
                    F[2, :, 0, :] = np.sqrt(self.dt) * B[i]
                    F[3, :, 1, :] = np.sqrt(self.dt) * B[i]
                    F[3, :, 0, :] = A[i, j]

                    Fexp = la.expm(F.reshape((8,8))).reshape((4, 2, 4, 2))[:, :, 0, :]

                    WA[i, j] = Fexp[3]
                    if j == 0:
                        WB[i] = Fexp[2]
                    if i == 0:
                        WC[j] = Fexp[1]

            site_tensor = np.zeros((A.shape[0]+1, A.shape[1]+1, 2, 2), dtype=complex)
            site_tensor[0, 0] = WD
            site_tensor[0, 1:] = WC
            site_tensor[1:, 0] = WB
            site_tensor[1:, 1:] = WA

            WII_list.append(site_tensor)

        D = W_list[-1][0]
        B = W_list[-1][1:-1]

        WD = la.expm(self.dt * D)
        WB = np.zeros(B.shape, dtype=complex)
        for i in range(WB.shape[0]):
            F = np.zeros((4, 2, 4, 2), dtype=complex)

            for k in range(4):
                F[k, :, k, :] = self.dt * D
            F[2, :, 0, :] = np.sqrt(self.dt) * B[i]
            F[3, :, 1, :] = np.sqrt(self.dt) * B[i]

            Fexp = la.expm(F.reshape((8,8))).reshape((4, 2, 4, 2))[:, :, 0, :]

            WB[i] = Fexp[2]

        site_tensor = np.zeros((B.shape[0]+1, 2, 2), dtype=complex)
        site_tensor[0] = WD
        site_tensor[1:] = WB

        WII_list.append(site_tensor)

        WII_list[0] = WII_list[0].transpose(1,2,0)
        for i in range(1, self.n_sites-1):
            WII_list[i] = WII_list[i].transpose(0,2,3,1)

        return WII_list


class MPSPQC:
    """
    Class containing the algorithm that transforms an MPS state into a PQC

    Args:
        - mps_list (np.array): list of tensors representing an MPS.

    Internal Methods:

        - left_mps_to_g_gates() -> np.array:
            Generates G-tensors from an MPS in left form.

        - without_truncation() -> np.array:
            Convert the MPS to left from untruncated.

        - with_canonical_truncation() -> np.array:
            Generates the truncation of the MPS by canonical
            truncation and convert the MPS to left from.

        - with_left_form_truncation() -> np.array:
            Generates the truncation of the MPS by the left form
            truncation and convert the MPS to left from.

        - reshaping_g_gates() -> np.array:
            Transforms four-legged G-tensors into 4x4 matrices
            following the correct contraction mapping

    Methods:
        - analytical_decomposition() -> list(np.array):
            Generate the set of G-tensors that reproduce the initial
            MPS using the analytical decomposition method.
    """

    def __init__(self, mps_list: np.array):
        self.mps_list = mps_list

    def _left_mps_to_g_gates(self, mps: np.array) -> np.array:
        # sourcery skip: merge-list-append, move-assign-in-block, switch, use-itertools-product
        """
        Generates G-tensors from an MPS in left form

        Args:
            - mps (np.array): the matrix product state

        Returns:

            - np.array: the unitary gates which encode the mps
        """

        n_capital = len(mps)
        d = 2
        g_list = []

        g_n = np.zeros((d, d, d, d), dtype=complex128)
        # Last tensor in orthogonality, must be normalized such that AA* = 1
        for k in range(d):
            for l in range(d):
                g_n[0, 0, k, l] = mps[-1][l][k][0]

        basis_matrix = np.conj(mps[-1][:, :].T).reshape(1, 4)
        kernel = null_space(basis_matrix)

        for num in range(d**2 - 1):
            if num == 1:
                i = 0
                j = 1
            elif num == 0:
                i = 1
                j = 0
            else:
                i = 1
                j = 1

            g_n[i, j, :, :] = kernel[:, num].reshape(2, 2)

        g_list.append(g_n)

        for n in range(2, n_capital):

            g_n = np.zeros((d, d, d, d), dtype=complex128)

            g_n[0, :, :, :] = mps[-n].transpose(1, 2, 0)
            kernel = null_space(np.conj(mps[-n].transpose(2, 0, 1).reshape(2, 4))).T

            g_n[1] = kernel.reshape(2, 2, 2).transpose(2, 0, 1)
            g_list.append(g_n)

        g_n = np.zeros((d, d), dtype=complex128)

        g_n = mps[-n_capital][0, :, :]

        g_list.append(g_n)

        return g_list

    def _without_truncation(self, mps: np.array) -> np.array:
        """
        Convert the MPS to left from untruncated.

        Args:
           - mps (np.array): represents the MPS to be converted into left form.

        Returns:
            np.array: represents the MPS in left form.
        """

        state_mps_left_form = TensorOperations(mps_list=mps).left_orthogonal_form()
        state_mps_left_form[-1] = state_mps_left_form[-1] / LA.norm(state_mps_left_form[-1])

        return state_mps_left_form

    def _with_canonical_truncation(self, mps: np.array, chi_max: int = 2) -> np.array:
        """
        Generates the truncation of the MPS by canonical truncation
        and convert the MPS to left from.

        Args:
           - mps (np.array): represents the MPS to be truncated.

        Returns:
            np.array: represents the truncated MPS.
        """

        tensor_mps_canonical = TensorOperations(mps_list=mps).canonical_network()

        tensor_mps_canonical_t = TensorOperations(mps_list=tensor_mps_canonical).truncation_canonical_network(
            chi=chi_max
        )

        tensor_mps_non_canonical_t = TensorOperations(mps_list=tensor_mps_canonical_t).mps_canonical_to_mps()

        return TensorOperations(mps_list=tensor_mps_non_canonical_t).truncation_network_left_form(chi=chi_max)

    def _with_left_form_truncation(self, mps: np.array, chi_max: int = 2) -> np.array:
        """
        Generates the truncation of the MPS by the left form truncation
        and convert the MPS to left from.

        Args:
           - mps (np.array): represents the MPS to be truncated.

        Returns:
            np.array: represents the truncated MPS.
        """
        return TensorOperations(mps_list=mps).truncation_network_left_form(chi=chi_max)

    def _reshaping_g_gates(self, g_gates_list: np.array) -> np.array:
        """
        Transforms four-legged G-tensors into 4x4 matrices
        following the correct contraction mapping

        Args:
            g_gates_list (np.array): list of 4x4 G tensioners.

        Returns:
            np.array: list of tensors in matrix format.
        """

        g_gates_list[1] = ncon([g_gates_list[0], g_gates_list[1]], [[-1, 1], [-2, -3, -4, 1]])

        g_gates_list_reshape = []

        for i in range(1, len(g_gates_list) - 1):

            g_gates_list[i] = g_gates_list[i].transpose(0, 2, 1, 3)
            g_gates_list_reshape.append(g_gates_list[i].reshape(4, 4))

        g_gates_list_reshape.append(g_gates_list[-1].reshape(4, 4))

        return g_gates_list_reshape

    def analytical_decomposition(
        self,
        mode: Union[str("canonical"), str("left"), str("non")] = "left",  # type: ignore
        k: Optional[int] = 10,
        f: Optional[float] = 0.8,
    ) -> np.array:
        """
        Generate the set of G-tensors that reproduce the initial MPS using
        the analytical decomposition method.

        Args:

            - k (int): the maximum amount of layers allowed for the
                resulting PQC (Parameterised Quantum Circuit).
            - f (int): the target fidelity.
            - mode (str): represents the method of truncation used.

        Returns:
            - list(np.array): the resulting PQC
        """

        if mode == "canonical":
            truncation_function = self._with_canonical_truncation
        elif mode == "left":
            truncation_function = self._with_left_form_truncation
        elif mode == "non":
            truncation_function = self._without_truncation

        if mode not in ["non", "left", "canonical"]:
            raise ValueError("Mode entered not valid")

        state_mps = self.mps_list.copy()
        n_sites = len(state_mps)

        chi_list = [self.mps_list[i].shape[2] for i in range(len(self.mps_list) - 1)]
        chi_value = list(filter(lambda x: x not in [1, 2], chi_list))
        chi_len_value = len(chi_value)

        if chi_len_value > 0 and mode == "non":
            raise ValueError("Incorrect mode with the MPS introduced")

        if n_sites <= 2:
            raise ValueError("A minimum of one MPS of three tensioners is required.")

        k_i = 0

        fidelidad = abs(TensorOperations(mps_list=state_mps).proyection_state(s="0" * n_sites)) ** 2

        circuit_layers = []

        while k_i < k and fidelidad < f:

            state_mps_t = truncation_function(mps=state_mps)

            state_mps_t[-1] = state_mps_t[-1] / LA.norm(state_mps_t[-1])

            g_gates_list = self._left_mps_to_g_gates(mps=state_mps_t)

            g_gates_list.reverse()

            g_gates_list_conj = [np.conj(g) for g in g_gates_list]

            tensor_ini = ncon(
                [
                    state_mps[0],
                    state_mps[1],
                    g_gates_list_conj[0],
                    g_gates_list_conj[1],
                ],
                [[-1, 1, 2], [2, 3, -4], [1, 4], [-2, 3, -3, 4]],
            )

            tensor_ini = tensor_ini.reshape(
                tensor_ini.shape[0],
                tensor_ini.shape[1],
                tensor_ini.shape[2] * tensor_ini.shape[3],
            )

            new_mps_state = [tensor_ini]

            if len(state_mps) > 3:

                for i in range(2, len(state_mps) - 1):

                    tensor_inter = ncon(
                        [state_mps[i], g_gates_list_conj[i]],
                        [[-1, 1, -5], [-3, 1, -4, -2]],
                    )
                    tensor_inter = tensor_inter.reshape(
                        tensor_inter.shape[0] * tensor_inter.shape[1],
                        tensor_inter.shape[2],
                        tensor_inter.shape[3] * tensor_inter.shape[4],
                    )
                    new_mps_state.append(tensor_inter)

            tensor_fin = ncon([state_mps[-1], g_gates_list_conj[-1]], [[-1, 1, -5], [-3, -4, 1, -2]])

            tensor_fin = tensor_fin.reshape(
                tensor_fin.shape[0] * tensor_fin.shape[1] * tensor_fin.shape[2],
                tensor_fin.shape[3] * tensor_fin.shape[4],
            )

            u_m, s_m, v_h = LA.svd(tensor_fin, full_matrices=False)

            l = len(s_m)

            u_l = u_m.reshape(new_mps_state[-1].shape[2], 2, l)

            u_r = ncon([np.diag(s_m), v_h], [[-1, 1], [1, -2]])

            u_r = u_r.reshape(l, 2, 1)

            new_mps_state.extend((u_l, u_r))

            new_mps_state_truncate = TensorOperations(mps_list=new_mps_state).truncation_network_left_form(chi=chi_list)

            new_mps_state_truncate[-1] = new_mps_state_truncate[-1] / LA.norm(new_mps_state_truncate[-1])

            g_gates_list_reshape = self._reshaping_g_gates(g_gates_list=g_gates_list)

            circuit_layers.append(g_gates_list_reshape)

            k_i += 1

            fidelidad = abs(TensorOperations(mps_list=new_mps_state_truncate).proyection_state(s="0" * n_sites)) ** 2

            state_mps = new_mps_state_truncate

        return circuit_layers, fidelidad

    def _generator_unitary_tensor(self, n_legs: int = 4, d: int = 2) -> np.array:
        """_summary_

        Args:
            n_legs (int, optional): _description_. Defaults to 4.
            d (int, optional): _description_. Defaults to 2.

        Returns:
            np.array: _description_
        """

        shape = (d,) * n_legs

        tensor = np.random.rand(*shape) + 1j * np.random.rand(*shape)

        tensor_matrix = tensor.reshape((d) * n_legs // 2, (d) * n_legs // 2)

        matrix_unitary, _ = np.linalg.qr(tensor_matrix)

        return matrix_unitary.reshape(shape)

    def _contract_left_environment_tensor(
        self, mps_one: np.array, mps_two: np.array, non_sites: Tuple[int, int]
    ) -> np.array:
        """_summary_

        Args:
            mps_one (np.array): _description_
            mps_two (np.array): _description_
            non_sites (int, int): _description_

        Returns:
            np.array: _description_
        """

        if min(non_sites) != 0:

            tensor = ncon([mps_one[0], mps_two[0]], [[1, 2, -1], [1, 2, -2]])

            if min(non_sites) != 1:

                for i in range(1, min(non_sites)):

                    tensor = ncon([tensor, mps_one[i], mps_two[i]], [[1, 2], [1, 3, -1], [2, 3, -2]])
        else:
            tensor = np.eye(1)

        return tensor

    def _contract_right_environment_tensor(
        self, mps_one: np.array, mps_two: np.array, non_sites: Tuple[int, int]
    ) -> np.array:
        """_summary_

        Args:
            mps_one (np.array): _description_
            mps_two (np.array): _description_
            non_sites (int, int): _description_

        Returns:
            np.array: _description_
        """

        if max(non_sites) != len(mps_one) - 1:

            tensor = ncon([mps_one[-1], mps_two[-1]], [[-1, 2, 1], [-2, 2, 1]])

            if max(non_sites) != len(mps_one) - 2:

                for i in range(len(mps_one) - 2, max(non_sites), -1):

                    tensor = ncon([mps_one[i], mps_two[i], tensor], [[-1, 3, 1], [-2, 3, 2], [1, 2]])
        else:

            tensor = np.eye(1)

        return tensor

    def _contract_environment_tensor(
        self, mps_one: np.array, mps_two: np.array, non_sites: Tuple[int, int]
    ) -> np.array:
        """_summary_

        Args:
            mps_one (np.array): _description_
            mps_two (np.array): _description_
            non_sites (int, int): _description_

        Raises:
            ValueError: _description_

        Returns:
            np.array: _description_
        """

        if len(mps_one) != len(mps_two):

            raise ValueError("The length of the MPS are different")

        left_rho = self._contract_left_environment_tensor(mps_one, mps_two, non_sites)

        right_rho = self._contract_right_environment_tensor(mps_one, mps_two, non_sites)

        part_f_one = ncon([mps_one[non_sites[0]], mps_one[non_sites[1]]], [[-1, -2, 1], [1, -3, -4]])

        part_f_two = ncon([mps_two[non_sites[0]], mps_two[non_sites[1]]], [[-1, -2, 1], [1, -3, -4]])

        return ncon(
            [left_rho, part_f_one, part_f_two, right_rho],
            [[1, 2], [1, -1, -2, 3], [2, -3, -4, 4], [3, 4]],
        )

    def _apply_all_over_mps(self, g_gates: np.array, mps: np.array, layers: int, gates_per_layer: int) -> np.array:
        """_summary_

        Args:
            g_gates (np.array): _description_
            mps (np.array): _description_
            layers (int): _description_

        Returns:
            np.array: _description_
        """

        g_gates_list = g_gates.copy()
        new_mps = mps.copy()

        for i, j in itertools.product(range(layers), range(gates_per_layer)):

            two_tensor_leg = ncon(
                [new_mps[j], new_mps[j + 1], np.conj(g_gates_list[j + gates_per_layer * i])],
                [[-1, 1, 2], [2, 3, -4], [1, 3, -2, -3]],
            )

            two_tensor_leg = two_tensor_leg.reshape(
                two_tensor_leg.shape[0] * two_tensor_leg.shape[1], two_tensor_leg.shape[2] * two_tensor_leg.shape[3]
            )

            utemp, sig_p, vhtemp = LA.svd(two_tensor_leg, full_matrices=False)

            t_l = utemp.reshape(new_mps[j].shape[0], 2, len(sig_p))

            vhtemp = vhtemp.reshape(len(sig_p), 2, new_mps[j + 1].shape[2])

            t_r = ncon([np.diag(sig_p), vhtemp], [[-1, 1], [1, -2, -3]])

            new_mps[j] = t_l
            new_mps[j + 1] = t_r

        return new_mps

    def _apply_gs_before(
        self, g_gates: np.array, mps: np.array, layers: int, gates_per_layer: int, n_split: int
    ) -> np.array:
        """_summary_

        Args:
            g_gates (np.array): _description_
            mps (np.array): _description_
            layers (int): _description_
            gates_per_layer (int): _description_
            n_split (int): _description_

        Returns:
            np.array: _description_
        """

        if n_split == 0:
            return mps

        g_gates_list = g_gates.copy()
        new_mps = mps.copy()

        order_list = list(itertools.product(range(layers), range(gates_per_layer)))
        order_list = order_list[:n_split]

        for i, j in order_list:

            two_tensor_leg = ncon(
                [new_mps[j], new_mps[j + 1], np.conj(g_gates_list[j + gates_per_layer * i])],
                [[-1, 1, 2], [2, 3, -4], [1, 3, -2, -3]],
            )

            two_tensor_leg = two_tensor_leg.reshape(
                two_tensor_leg.shape[0] * two_tensor_leg.shape[1], two_tensor_leg.shape[2] * two_tensor_leg.shape[3]
            )

            utemp, sig_p, vhtemp = LA.svd(two_tensor_leg, full_matrices=False)

            t_l = utemp.reshape(new_mps[j].shape[0], 2, len(sig_p))

            vhtemp = vhtemp.reshape(len(sig_p), 2, new_mps[j + 1].shape[2])

            t_r = ncon([np.diag(sig_p), vhtemp], [[-1, 1], [1, -2, -3]])

            new_mps[j] = t_l
            new_mps[j + 1] = t_r

        return new_mps

    def _apply_gs_after(
        self, g_gates: np.array, mps: np.array, layers: int, gates_per_layer: int, n_split: int
    ) -> np.array:
        """_summary_

        Args:
            g_gates (np.array): _description_
            mps (np.array): _description_
            layers (int): _description_
            gates_per_layer (int): _description_
            n_split (int): _description_

        Returns:
            np.array: _description_
        """

        if n_split == len(g_gates) - 1:
            return mps

        g_gates_list = g_gates.copy()
        new_mps = mps.copy()

        order_list = list(itertools.product(range(layers), range(gates_per_layer)))
        order_list = order_list[n_split + 1 :]
        order_list.reverse()

        for i, j in order_list:

            two_tensor_leg = ncon(
                [new_mps[j], new_mps[j + 1], g_gates_list[j + gates_per_layer * i]],
                [[-1, 1, 2], [2, 3, -4], [-2, -3, 1, 3]],
            )

            two_tensor_leg = two_tensor_leg.reshape(
                two_tensor_leg.shape[0] * two_tensor_leg.shape[1], two_tensor_leg.shape[2] * two_tensor_leg.shape[3]
            )

            utemp, sig_p, vhtemp = LA.svd(two_tensor_leg, full_matrices=False)

            t_l = utemp.reshape(new_mps[j].shape[0], 2, len(sig_p))

            vhtemp = vhtemp.reshape(len(sig_p), 2, new_mps[j + 1].shape[2])

            t_r = ncon([np.diag(sig_p), vhtemp], [[-1, 1], [1, -2, -3]])

            new_mps[j] = t_l
            new_mps[j + 1] = t_r

        return new_mps

    def _new_g_operator(self, f_tensor: np.array, u_m_old: np.array, ratio: float) -> np.array:
        """_summary_

        Args:
            f_tensor (np.array): _description_
            u_m_old (np.array): _description_
            ratio (float): _description_

        Returns:
            np.array: _description_
        """

        f_tensor = f_tensor.reshape(4, 4)
        u_m_old = u_m_old.reshape(4, 4)

        utemp, _, vhtemp = LA.svd(f_tensor, full_matrices=False)
        u_m_new = utemp @ vhtemp

        u_inter = np.conj(u_m_old.T) @ u_m_new

        eigenvalues, eigenvectors = np.linalg.eig(u_inter)

        u_inter_fin = np.dot(eigenvectors, np.dot(np.diag(eigenvalues**ratio), np.linalg.inv(eigenvectors)))

        u_m_p = u_m_old @ u_inter_fin

        # TODO: check if is correct this step

        utemp, _, vhtemp = LA.svd(u_m_p, full_matrices=False)
        u_m_p = utemp @ vhtemp

        u_m_p = u_m_p.reshape(2, 2, 2, 2)

        return u_m_p

    def _gates_g_from_g_tensors(self, list_g_tensors: np.array, gates_per_layer: int) -> np.array:
        """_summary_

        Args:
            list_g_tensors (np.array): _description_

        Returns:
            np.array: _description_
        """

        list_g_gates = [list_g_tensors[i].reshape(4, 4) for i in range(len(list_g_tensors))]

        list_of_list_g = [list_g_gates[i : i + gates_per_layer] for i in range(0, len(list_g_gates), gates_per_layer)]

        for j, layer_gates in enumerate(list_of_list_g):
            for i, gate in enumerate(layer_gates):

                list_of_list_g[j][i] = gate.transpose(1, 0)

        return list_of_list_g

    def optimization_decomposition(
        self, f: float = 0.8, sweeps: int = 10, list_g_tensors: np.array = None, layers: int = 1, ratio: float = 0.6
    ) -> np.array:
        """_summary_

        Args:
            f (float, optional): _description_. Defaults to 0.8.
            sweeps (int, optional): _description_. Defaults to 10.
            list_g_tensors (np.array, optional): _description_. Defaults to None.
            layers (int, optional): _description_. Defaults to 1.
            ratio (float, optional): _description_. Defaults to 0.6.

        Raises:
            ValueError: _description_

        Returns:
            np.array: _description_
        """

        if not list_g_tensors:

            list_g_tensors = [
                self._generator_unitary_tensor(n_legs=4, d=2) for _ in range(layers * (len(self.mps_list) - 1))
            ]

        else:

            list_g_tensors = [elemento for sublist in list_g_tensors for elemento in sublist]
            list_g_tensors = [list_g_tensors[i].reshape(2, 2, 2, 2) for i in range(len(list_g_tensors))]

            layers = len(list_g_tensors) // (len(self.mps_list) - 1)

        if len(list_g_tensors) % (len(self.mps_list) - 1) != 0:

            raise ValueError("Length of MPS or G tensioners not correct")

        n_sites = len(self.mps_list)

        gates_per_layer = n_sites - 1

        zeros_state = MPSGenerator(d=2, chi=1, n_sites=n_sites).zeros_mps()

        new_mps_ini = self._apply_all_over_mps(
            g_gates=list_g_tensors, mps=zeros_state, layers=layers, gates_per_layer=gates_per_layer
        )

        fidelity = abs(TensorOperations(mps_list=self.mps_list).proyection_mps_operator(mps_bra=new_mps_ini)) ** 2
        t = 0

        fidelity_list = [fidelity]

        while fidelity < f and t < sweeps:

            for i, j in itertools.product(range(layers), range(gates_per_layer)):

                non_sites = (j, j + 1)

                n_split = j + gates_per_layer * i

                mps_one = self._apply_gs_before(
                    g_gates=list_g_tensors,
                    mps=zeros_state,
                    layers=layers,
                    gates_per_layer=gates_per_layer,
                    n_split=n_split,
                )
                mps_two = self._apply_gs_after(
                    g_gates=list_g_tensors,
                    mps=self.mps_list,
                    layers=layers,
                    gates_per_layer=gates_per_layer,
                    n_split=n_split,
                )

                f_tensor = self._contract_environment_tensor(mps_one=mps_one, mps_two=mps_two, non_sites=non_sites)
                u_m_old = list_g_tensors[n_split]

                list_g_tensors[n_split] = self._new_g_operator(f_tensor=f_tensor, u_m_old=u_m_old, ratio=ratio)

            new_mps_ini = self._apply_all_over_mps(
                g_gates=list_g_tensors, mps=zeros_state, layers=layers, gates_per_layer=gates_per_layer
            )

            fidelity = abs(TensorOperations(mps_list=self.mps_list).proyection_mps_operator(mps_bra=new_mps_ini)) ** 2

            t += 1

            fidelity_list.append(fidelity)

        list_g_gates = self._gates_g_from_g_tensors(list_g_tensors=list_g_tensors, gates_per_layer=gates_per_layer)

        return list_g_gates, fidelity_list


class Pretraining:
    """_summary_"""

    def __init__(self,
                 ham,
                 n_qubits: int,
                 chi_mpo: int,
                 fidelity_mps: float = 0.95,
                 sweeps_mps: int = 10,
                 init_layers_mps: int = 1, 
                 layers_su4: int = None,
                 connectivity_su4: Optional[Union[str("Circular"), str("Linear"), str("Full"), str("ExtendFull"), List[Tuple[int, int]]]] = "ExtendFull",  # type: ignore
                 seed=None):

        self.ham = ham
        self.n_qubits = n_qubits
        self.chi_mpo = chi_mpo
        self.fidelity_mps = fidelity_mps
        self.sweeps_mps = sweeps_mps
        self.init_layers_mps = init_layers_mps
        self.layers_su4 = layers_su4
        self.connectivity_su4 = connectivity_su4
        self.seed = seed

    def _construct_extended_circuit(self, n_qubits: int, gates: np.ndarray):
        """_summary_

        Args:
            n_qubits (int): _description_
            gates (np.ndarray): _description_

        Returns:
            _type_: _description_
        """

        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        circuit = QuantumCircuit(qr, cr)
        for j, layer_gates in enumerate(gates):
            for i, gate in enumerate(layer_gates):
                gate_decomp = KAKDecomposition(gate)
                circuit, _ = gate_decomp.add_to_circuit(circuit, (i, i + 1))

                if j == len(gates) - 1:
                    for k in range(i + 2, n_qubits):
                        unitary_gate = np.identity(4)
                        gate_decomp = KAKDecomposition(unitary_gate)
                        circuit, _ = gate_decomp.add_to_circuit(circuit, (i, k))

        # for i in range(n_qubits):
        #     circuit.append(gates.Measure(), [i], [i])

        return circuit

    def _construct_mps_circuit(self, n_qubits: int, gates_optimization: np.array):
        """_summary_

        Args:
            n_qubits (int): _description_
            gates_optimization (np.array): _description_

        Returns:
            _type_: _description_
        """

        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        circuit = QuantumCircuit(qr, cr)

        for _, layer_gates in enumerate(gates_optimization):
            for i, gate in enumerate(layer_gates):
                gate_decomp = KAKDecomposition(gate)
                circuit, _ = gate_decomp.add_to_circuit(circuit, (i, i + 1))

        # for i in range(n_qubits):
        #     circuit.append(gates.Measure(), [i], [i])

        return circuit

    def _construct_su4_circuit(
        self,
        n_qubits: int,
        layers_su4: int,
        connectivity_su4: Union[
            "Circular",  # type: ignore
            "Linear",  # type: ignore
            "Full",  # type: ignore
            List[Tuple[int, int]],
        ] = "Linear",
    ):  # type: ignore

        if layers_su4:

            if isinstance(connectivity_su4, list):
                connectivity = connectivity_su4
            elif connectivity_su4 == "Full":
                connectivity = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
            else:
                connectivity = [(i, i + 1) for i in range(n_qubits - 1)] + (
                    [(n_qubits - 1, 0)] if connectivity_su4 == "Circular" else []
                )

            qr = QuantumRegister(n_qubits)
            cr = ClassicalRegister(n_qubits)
            circuit_su4 = QuantumCircuit(qr, cr)

            for i in range(layers_su4):
                for i, j in connectivity:

                    gate_decomp_i = KAKDecomposition(np.identity(4))

                    circuit_su4, _ = gate_decomp_i.add_to_circuit(circuit_su4, (i, j))

            # for i in range(n_qubits):
            #     circuit_su4.append(gates.Measure(), [i], [i])

        else:

            circuit_su4 = None

        return circuit_su4

    def dmrg_pretraining(
        self,
        metrics: bool = False,
        prep_state_dmrg: Union[str("Decay_state"), str("Uniform_state"), str("Two_state"), str("One_state"), str("Gibbs_state")] = "One_state",  # type: ignore
        sweeps_dmrg: int = 3,
        lanczos_maxit: int = 2,
        lanczos_krydim: int = 4,
    ):
        """_summary_

        Args:
            ham (_type_): _description_
            n_qubits (int): _description_
            chi (int): _description_
            sweeps_dmrg (int, optional): _description_. Defaults to 3.
            lanczos_maxit (int, optional): _description_. Defaults to 2.
            lanczos_krydim (int, optional): _description_. Defaults to 4.
            f_mps (float, optional): _description_. Defaults to 0.95.
            k_mps (int, optional): _description_. Defaults to 1.
            sweeps_mps (int, optional): _description_. Defaults to 10.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.n_qubits % 2 == 1:
            raise NotImplementedError("The number of qubits must be even")

        ham_mpo = MPO(n_sites=self.n_qubits, phys_d=2, chi=self.chi_mpo, open_bounds=False)
        ham_mpo.from_ham(self.ham)

        dmrg_instance = DMRG(
            mpo=ham_mpo,
            chi=self.chi_mpo,
            sweeps=sweeps_dmrg,
            lanczos_maxit=lanczos_maxit,
            lanczos_krydim=lanczos_krydim,
            prep_state=prep_state_dmrg,
            metrics=metrics,
        )

        _, rightortho_ground_mps, _, dmrg_energies, coherence, effective_dimension = dmrg_instance.run()

        gates_optimization, fidelity = MPSPQC(mps_list=rightortho_ground_mps.tensors).optimization_decomposition(
            sweeps=self.sweeps_mps, layers=self.init_layers_mps, f=self.fidelity_mps, ratio = 0.4
        )

        if self.connectivity_su4 == "ExtendFull":

            circuit_mps = self._construct_extended_circuit(n_qubits=self.n_qubits, gates=gates_optimization)
            circuit_su4 = None

        else:

            circuit_mps = self._construct_mps_circuit(n_qubits=self.n_qubits, gates_optimization=gates_optimization)
            circuit_su4 = self._construct_su4_circuit(
                n_qubits=self.n_qubits, layers_su4=self.layers_su4, connectivity_su4=self.connectivity_su4
            )

        return StructureResultsPretraining(
            effective_dimension=effective_dimension,
            coherence=coherence,
            fun=dmrg_energies[-1],
            fun_history=dmrg_energies,
            fidelity=fidelity[-1],
            fidelity_history=fidelity,
            circuit_mps=circuit_mps,
            circuit_su4=circuit_su4,
        )
        
    def time_evolution_pretraining(self, 
                                   n_steps,
                                   dt: float,
                                   order: int = 1,
                                   metrics: bool = True,
                                   init_mps = None,
                                   stop_energy: float = None
                                   ):
        
        if self.seed is not None:
            np.random.seed(self.seed)

        if self.n_qubits % 2 == 1:
            raise NotImplementedError("The number of qubits must be even")

        tevo = TimeEvolution(n_sites = self.n_qubits, 
                      ham = self.ham, 
                      init_mps = init_mps, 
                      chi = self.chi_mpo,
                      stop_energy = stop_energy)
        
        final_mps = tevo.run(n_steps = n_steps, dt = dt, order = order, metrics = metrics)
        
        gates_optimization, fidelity = MPSPQC(mps_list=final_mps.tensors).optimization_decomposition(
            sweeps=self.sweeps_mps, layers=self.init_layers_mps, f=self.fidelity_mps
        )

        if self.connectivity_su4 == "ExtendFull":

            circuit_mps = self._construct_extended_circuit(n_qubits=self.n_qubits, gates=gates_optimization)
            circuit_su4 = None

        else:

            circuit_mps = self._construct_mps_circuit(n_qubits=self.n_qubits, gates_optimization=gates_optimization)
            circuit_su4 = self._construct_su4_circuit(
                n_qubits=self.n_qubits, layers_su4=self.layers_su4, connectivity_su4=self.connectivity_su4
            )

        return StructureResultsPretraining(
            effective_dimension=tevo.effective_dimension,
            coherence=tevo.coherence,
            fun=tevo.energies[-1],
            fun_history=tevo.energies,
            fidelity=fidelity[-1],
            fidelity_history=fidelity,
            circuit_mps=circuit_mps,
            circuit_su4=circuit_su4,
        )


class SamplingMPS:
    """
    Class that implements the perfect sampling algorithm
    
    Internal Methods:
    
        - _density_matrix() -> np.array:
            Creates the density matrix from which the pb measurements are obtained.
        - _obtain_state_i() -> int:
            Establishes that the qubit of position i is in state 0 or 1.
        - _proyecting_state() -> np.array:
            Creates the state 0 or 1 in which the MPS tensor of the i.
        - _proyect_state_i() -> np.array:
            Projects the state of position i onto the MPS.
        - _right_envs() -> list(np.array):
            Calculates the list of contracted tensors needed to obtain the density matrix.
        - _one_sample() -> np.array:
            Implements a single sampling of the MPS network.
            
    Methods:

        - sampling() -> dict(str, float)):
            Implements MPS state sampling.

    """
    def __init__(self): ...
    
    def _density_matrix(
        self,
        mps: np.array,
        r_site: int,
    ) -> np.array:
        """
        Creates the density matrix from which the pb measurements are obtained.
        Args:
            mps (np.array): MPS to sample.
            r_site (int): position into the MPS.

        Returns:
            np.array: the density matrix.
        """

        tensor_right = self.tensor_right[r_site + 1]

        return ncon([mps[0], np.conj(mps[0]), tensor_right], [[1, -1, 2], [1, -2, 3], [2, 3]])

    def _obtain_state_i(
        self,
        mps: np.array,
        r_site: int,
    ) -> int:
        """
        Establishes that the qubit of position i is in state 0 or 1.
        
        Args:
            mps (np.array): MPS for sampling
            r_site (int): the position into the MPS for establishes the value of the state. 

        Returns:
            int: the value of the final state 0 or 1 for the qubit in the position i.
        """

        reduce_matrix = self._density_matrix(mps=mps, r_site=r_site)

        pb_values = list(np.real(np.diag(reduce_matrix)))

        return (0, pb_values[0]) if random.uniform(0, 1) < pb_values[0] else (1, pb_values[1])

    def _proyecting_state(self, tensor: np.array, state: int) -> np.array:
        """
        Creates the state 0 or 1 in which the MPS tensor of the i.

        Args:
            tensor (np.array): the tensor on which we make the projection.
            state (int): the value of the state.

        Returns:
            np.array: the tensor that represents the state 0 or 1
        """

        state_final = np.array([[1, 0]]) if state == 0 else np.array([[0, 1]])

        return ncon([state_final, tensor], [[-1, 1], [1, -2]])

    def _proyect_state_i(self, mps: np.array, r_site: int): 
        """
        Projects the state of position i onto the MPS.

        Args:
            mps (np.array): MPS for sampling.
            r_site (int): the position into the MPS for establishes the value of the state. 

        Returns:
            - np.array: the final tensor on the position i.
            - int: the value of the tensor on the position i.
        """

        if len(mps) > 1:

            mps_new = mps.copy()
            mps_new = mps_new[1:]

        else:
            mps_new = mps.copy()

        state_qubit, pb_state = self._obtain_state_i(mps=mps, r_site=r_site)

        state_to_proyect = mps[0].reshape(2, mps[0].shape[2])

        state_proyected = self._proyecting_state(tensor=state_to_proyect, state=state_qubit)

        state_proyected = state_proyected.reshape(1, mps[0].shape[2])

        mps_new[0] = ncon([state_proyected, mps_new[0]], [[-1, 1], [1, -2, -3]])

        mps_new[0] = 1 / np.sqrt(pb_state) * mps_new[0].reshape(1, 2, mps_new[0].shape[2])

        return mps_new, state_qubit

    def _one_sample(self, mps: np.array):
        """
        Implements a single sampling of the MPS network.

        Args:
            mps (np.array): MPS for sampling.

        Returns:
            np.array: _description_
        """

        mps_sample = mps.copy()

        sample = []

        for i in range(len(mps_sample)):

            mps_sample, state_qubit = self._proyect_state_i(mps=mps_sample, r_site=i)

            sample.append(state_qubit)

        return sample

    def _right_envs(self, mps: np.array):
        """
        Calculates the list of contracted tensors needed to obtain the density matrix.

        Args:
            mps (np.array): MPS to sampling.

        Returns:
            list(np.array()): list of density matrix.
        """

        ortho_renv = [None for _ in range(len(mps))]

        ortho_renv[-1] = ncon([mps[-1], np.conj(mps[-1])], [[-1, 1, 2], [-2, 1, 2]])

        for s in range(len(mps) - 2, -1, -1):

            ortho_renv[s] = ncon(
                [mps[s], np.conj(mps[s]), ortho_renv[s + 1]],
                [[-1, 1, 2], [-2, 1, 3], [2, 3]],
            )

        ortho_renv.append(np.eye(1))

        return ortho_renv

    def sampling(self, mps: np.array, n_samples: int):
        """
        Implements MPS state sampling.

        Args:
            mps (np.array): MPS for sampling.
            n_samples (int): number of samples.

        Returns:
            dict(str, float)): the dict that contains the results (state, probability)
        """

        self.tensor_right = self._right_envs(mps=mps)

        list_samples = [self._one_sample(mps=mps) for _ in range(n_samples)]

        counting = defaultdict(int)

        for sub_list in list_samples:
            key = "".join(map(str, sub_list))
            counting[key] += 1

        return {key: count / n_samples for key, count in counting.items()}
