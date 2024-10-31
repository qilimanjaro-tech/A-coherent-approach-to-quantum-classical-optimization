from typing import Optional, Union, List
import numpy as np
from numpy import complex128
from numpy import linalg as LA
from ncon import ncon

from variational_algorithms.hamiltonians import Hamiltonian


def _left_orthogonalize_tensor(tensor: np.ndarray, dtol=1e-12) -> (np.ndarray, np.ndarray):  # type: ignore
    """
    Left orthogonalizes a given tensor from an MPS.

    Args:
        - tensor (np.ndarray): The input tensor to be left orthogonalized.
        - dtol (float, optional): The tolerance for determining non-zero eigenvalues. Defaults to 1e-12.

    Returns:
        - np.ndarray: The left orthonormalized tensor.
        - np.ndarray: The matrix used for orthogonalizaion that must be plugged into the tensor to the right of the input tensor in the MPS.
    """

    rho = ncon([tensor, np.conj(tensor)], [[1, 2, -1], [1, 2, -2]])
    dtemp, utemp = LA.eigh(rho)

    # take the non-zero eigenvalues in decreasing order
    chitemp = sum(dtemp > dtol)
    d = dtemp[range(-1, -chitemp - 1, -1)]
    u = utemp[:, range(-1, -chitemp - 1, -1)]

    # the square root of the eigenvalues
    sq_d = np.sqrt(abs(d))

    # X is the square root of rho and Xinv is the inverse of X
    X = np.conj(u) @ np.diag(sq_d) @ u.T
    Xinv = np.conj(u) @ np.diag(1 / sq_d) @ u.T

    # we merge Xinv to the tensor such that it becomes leftorthogonal
    newtensor = ncon([tensor, Xinv], [[-1, -2, 1], [1, -3]])

    # the matrix X will be plugged to the right neighbor
    return newtensor, X


def _right_orthogonalize_tensor(tensor: np.ndarray, dtol=1e-12) -> (np.ndarray, np.ndarray):  # type: ignore
    """
    Right orthogonalizes a given tensor from an MPS.

    Args:
        - tensor (np.ndarray): The input tensor to be left orthogonalized.
        - dtol (float, optional): The tolerance for determining non-zero eigenvalues. Defaults to 1e-12.

    Returns:
        - np.ndarray: The right orthogonalized tensor.
        - np.ndarray: The matrix used for orthogonalizaion that must be plugged into the tensor to the left of the input tensor in the MPS.
    """

    rho = ncon([tensor, np.conj(tensor)], [[-1, 1, 2], [-2, 1, 2]])
    dtemp, utemp = LA.eigh(rho)

    # take the non-zero eigenvalues in decreasing order
    chitemp = sum(dtemp > dtol)
    d = dtemp[range(-1, -chitemp - 1, -1)]
    u = utemp[:, range(-1, -chitemp - 1, -1)]

    # the square root of the eigenvalues
    sq_d = np.sqrt(abs(d))

    # X os the square root of rho and Xinv is the inverse of X
    X = np.conj(u) @ np.diag(sq_d) @ u.T
    Xinv = np.conj(u) @ np.diag(1 / sq_d) @ u.T

    # we merge Xinv to the tensor such that it becomes rightorthogonal
    newtensor = ncon([Xinv, tensor], [[-1, 1], [1, -2, -3]])

    # the matrix X will be plugged to the left neighbor
    return newtensor, X


class TensorTrain:
    """
    Class representing a 1-dimensional tensor network.

    Args:
        - n_sites (int): The number of sites in the tensor network.
        - phys_d (int): The physical dimension of each site.
        - chi (int): The maximum bond dimension of the tensor train.
        - open_bounds (bool, optional): Whether the tensor train has open boundary conditions. Defaults to True.

    Raises:
        - NotImplementedError: If closed boundary conditions are specified.
    """

    def __init__(self, n_sites: int, phys_d: int, chi: int, open_bounds: bool = True):
        self.n_sites = n_sites
        self.d = phys_d
        self.chi = chi
        self.bounds = open_bounds
        self._tensors = [None] * n_sites

    def __getitem__(self, key: int) -> np.ndarray:
        """
        Get the tensor at the specified index.

        Args:
            - key (int): The index of the tensor to retrieve.

        Returns:
            - np.ndarray: The tensor at the specified index.
        """

        return self._tensors[key]


class MPS(TensorTrain):
    """
    Class representing a matrix product state (MPS) tensor network.

    Args:
        - n_sites (int): The number of sites in the tensor network.
        - phys_d (int): The physical dimension of each site.
        - chi (int): The maximum bond dimension of the MPS.
        - open_bounds (bool, optional): Whether the MPS has open boundary conditions. Defaults to True.

    Attributes:
        - tensors (List[np.ndarray]): The list of tensors in the tensor network.
        - n_sites (int): The number of sites in the tensor network.
        - d (int): The physical dimension of each site.
        - chi (int): The maximum bond dimension of the MPS.
        - bounds (bool): Whether the MPS has open boundary conditions.

    Methods:
        - random_init(max_bond: bool = False): Randomly initializes the tensor network.
        - norm() -> complex: Calculate the norm of the tensor network.
        - normalize(): Normalize the tensor network by dividing each tensor by the square root of the norm of the MPS.
        - left_orthogonalize(sites: Optional[int], dtol: float = 1e-12): Bring the MPS into left orthogonal form.
        - right_orthogonalize(sites: Optional[int], dtol: float = 1e-12): Bring the MPS into right orthogonal form.
        - apply_mpo(mpo: MPO): Apply an MPO to the MPS.

    """

    def __init__(self, n_sites: int, phys_d: int, chi: int, open_bounds: bool = True):
        if not open_bounds:
            raise NotImplementedError("Only open boundary conditions are currently supported for MPS's")
        super().__init__(n_sites, phys_d, chi, open_bounds)

    @property
    def tensors(self) -> List[np.ndarray]:
        """
        Returns the list of tensors in the tensor network.

        Returns:
            - List[np.ndarray]: The list of tensors in the tensor network.

        """

        return self._tensors

    @tensors.setter
    def tensors(self, tensors: List[np.ndarray]) -> None:
        """
        Set the tensors of the tensor network.

        Args:
            - tensors (List[np.ndarray]): List of rank-3 tensors representing the tensors of the tensor network.

        Raises:
            - ValueError: If the number of tensors is not equal to the number of sites, if any tensor is not a rank-3 tensor,
                if the tensor dimension is not equal to the physical dimension or if the tensor bond dimension is greater than
                the specified maximum bond dimension.
            - Warning: If the right bond dimension of a tensor is not equal to the left bond dimension of the next tensor.

        """

        if len(tensors) != self.n_sites:
            raise ValueError("Number of tensors must equal number of sites")
        for i in range(len(tensors)):
            if len(tensors[i].shape) != 3:
                raise ValueError("Tensor must be a rank-3 tensor")
            if tensors[i].shape[1] != self.d:
                raise ValueError(f"Tensor dimension must equal physical dimension {self.d}")
            if tensors[i].shape[0] > self.chi or tensors[i].shape[2] > self.chi:
                raise ValueError(f"Tensor bond dimension must be less than or equal to {self.chi}")
            # sourcery skip: merge-nested-ifs
            if i != len(tensors) - 1:
                if tensors[i].shape[2] != tensors[i + 1].shape[0]:
                    raise Warning(
                        f"Right bond dimension of tensor {i} must be equal to left bond dimension of tensor {i + 1} to be able to contract the MPS."
                    )

        self._tensors = tensors

    def __setitem__(self, key: int, tensor: np.ndarray) -> None:
        # sourcery skip: merge-else-if-into-elif
        """
        Set the tensor at the specified key index in the MPS.

        Args:
            - key (int): The index of the tensor.
            - tensor (np.ndarray): The tensor to be set.

        Raises:
            - ValueError: If the index is out of bounds, the tensor is not a rank-3 tensor,
                or the tensor dimensions do not match the expected values.
            - Warning: If the tensor dimensions do not match the neighboring tensors' dimensions.

        """

        if key < 0 or key >= self.n_sites:
            raise ValueError("Index out of bounds")

        if len(tensor.shape) != 3:
            raise ValueError("Tensor must be a rank-3 tensor")
        if tensor.shape[1] != self.d:
            raise ValueError(f"Tensor physical dimension must equal to {self.d}.")
        if tensor.shape[0] > self.chi or tensor.shape[2] > self.chi:
            raise ValueError(f"Tensor bond dimension must be less than or equal to {self.chi}")

        # TODO Check if this will restrict updating the tensors on DMRG sweeps when increasing bond dimension

        if key == 0:
            if tensor.shape[2] != self._tensors[key + 1].shape[0]:
                raise Warning(
                    f"Right bond dimension must be equal to {self._tensors[key + 1].shape[0]} to be able to contract the MPS."
                )

        elif key == self.n_sites - 1:
            if tensor.shape[0] != self._tensors[key - 1].shape[2]:
                raise Warning(
                    f"Left bond dimension must be equal to {self._tensors[key - 1].shape[2]} to be able to contract the MPS."
                )
        else:
            if tensor.shape[0] != self._tensors[key - 1].shape[2] or tensor.shape[2] != self._tensors[key + 1].shape[0]:
                raise Warning(
                    f"Left bond dimension must be equal to {self._tensors[key - 1].shape[2]} and right bond dimension must be equal to {self._tensors[key + 1].shape[0]} to be able to contract the MPS."
                )
        self._tensors[key] = tensor

    def random_init(self, max_bond: bool = False, complex_values: bool = False):
        """
        Randomly initializes the tensor network.

        Args:
            - max_bond (bool, optional): If True, initializes the tensor network with with all bond dimensions equal to the maximum bond dimension.
                Defaults to False.

        Raises:
            - NotImplementedError: If complex_values is True.

        """

        if complex_values:
            raise NotImplementedError("Complex values are not currently supported")

        if max_bond:
            self._tensors[0] = np.random.rand(1, self.d, self.chi)
            for i in range(1, self.n_sites - 1):
                self._tensors[i] = np.random.rand(self.chi, self.d, self.chi)

            self._tensors[-1] = np.random.rand(self.chi, self.d, 1)
        else:
            self._tensors[0] = np.random.rand(1, self.d, min(self.chi, 2))

            for i in range(1, self.n_sites - 1):
                self._tensors[i] = np.random.rand(
                    min(self.chi, min(2 ** (i), 2 ** (self.n_sites - i))),
                    self.d,
                    min(self.chi, min(2 ** (i + 1), 2 ** (self.n_sites - i - 1))),
                )
            self._tensors[-1] = np.random.rand(min(self.chi, 2), self.d, 1)

    def ones_init(self, max_bond: bool = False):
        if max_bond:
            self._tensors[0] = np.ones((1, self.d, self.chi))
            for i in range(1, self.n_sites - 1):
                self._tensors[i] = np.ones((self.chi, self.d, self.chi))

            self._tensors[-1] = np.ones((self.chi, self.d, 1))
        else:
            self._tensors[0] = np.ones((1, self.d, min(self.chi, 2)))

            for i in range(1, self.n_sites - 1):
                self._tensors[i] = np.ones(
                    (
                        min(self.chi, min(2 ** (i), 2 ** (self.n_sites - i))),
                        self.d,
                        min(self.chi, min(2 ** (i + 1), 2 ** (self.n_sites - i - 1))),
                    )
                )
            self._tensors[-1] = np.ones((min(self.chi, 2), self.d, 1))

    def _tensor_shapes(self, site: int):
        d = self.d
        if site == 0:
            return (1, d, int(min(d, self.chi)))

        elif site == self.n_sites - 1:
            return (int(min(d, self.chi)), d, 1)

        else:
            return (
                self._tensors[site - 1].shape[2],
                d,
                min(self.chi, self._tensors[site - 1].shape[2] * d, d ** (self.n_sites - site - 1)),
            )

    def empty_mps(self):
        self._tensors = np.ndarray(self.n_sites, dtype=object)

        for i in range(self.n_sites):
            self._tensors[i] = np.zeros(self._tensor_shapes(i))

    def state_init(self, state: str, open_bounds: bool = False):
        if len(state) != self.n_sites:
            raise ValueError("The length of the state string must be equal to the number of sites in the MPS")

        if any(int(state[i]) > self.d for i in range(self.n_sites)):
            raise ValueError(
                "The state string contains a site of higher dimension than the physical dimension of the MPS"
            )
        self.empty_mps()

        if open_bounds:
            self._tensors[0][0, int(state[0]), 0] = 1
            self._tensors[-1][0, int(state[-1]), 0] = 1
        else:
            self._tensors[0][int(state[0]), 0] = 1
            self._tensors[-1][0, int(state[-1])] = 1

        for j in range(1, self.n_sites - 1):
            self._tensors[j][0, int(state[j]), 0] = 1

    def norm(self) -> complex:
        """
        Calculate the norm of the tensor network.

        Returns:
            - complex: The norm of the tensor network.
        """

        S = ncon([self._tensors[0], np.conj(self._tensors[0])], [[1, 2, -1], [1, 2, -2]])
        for i in range(1, self.n_sites - 1):
            tensor = self._tensors[i]
            S = ncon([S, tensor, np.conj(tensor)], [[1, 2], [1, 3, -1], [2, 3, -2]])
        S = ncon([S, self._tensors[-1], np.conj(self._tensors[-1])], [[1, 2], [1, 3, 4], [2, 3, 4]])
        return S

    def normalize(self):
        """
        Normalize the tensor network by dividing each tensor by the square root of the norm of the MPS.
        """

        norm = self.norm()
        for i in range(self.n_sites):
            self._tensors[i] /= (norm) ** (1 / (2 * self.n_sites))

    def left_orthogonalize(self, sites: Optional[int] = None, dtol: float = 1e-12, normalize: bool = True):
        """
        Bring the MPS into left orthogonal form by left orthogonalizing the tensors, starting from the leftmost tensor and moving right sites positions.

        Args:
            -  (int, optional): Number of sites of the MPS to bring to left orthogonal form. If not specified, all tensors up to the second to last one are left orthogonalized.
            - dtol (float, optional):  The tolerance for determining non-zero eigenvalues in the SVD process done in the orthogonalization. Defaults to 1e-12.
            - normalize (bool, optional): Whether to normalize the MPS after orthogonalization. Defaults to True.


        Raises:
            - ValueError: If sites is an integer bigger than the number of sites minus one.
        """

        if sites is None:
            nstop = self.n_sites - 1
        elif isinstance(sites, int) and sites < self.n_sites:
            nstop = sites
        else:
            raise ValueError("Given sites must be an integer less than the number of sites minus one. ")

        for i in range(nstop):
            self._tensors[i], X = _left_orthogonalize_tensor(self._tensors[i], dtol)
            self._tensors[i + 1] = ncon([X, self._tensors[i + 1]], [[-1, 1], [1, -2, -3]])

        if normalize:
            norm = ncon([self._tensors[-1], np.conj(self._tensors[-1])], [[1, 2, 3], [1, 2, 3]])
            self._tensors[-1] /= np.sqrt(norm)

    def right_orthogonalize(self, sites: Optional[int] = None, dtol=1e-12, normalize: bool = True):
        """
        Bring the MPS into right orthogonal form by right orthogonalizing the tensors, starting from the rightmost tensor and moving left sites positions.

        Args:
            - sites (int, optional): Number of sites of the MPS to bring to left orthogonal form. If not specified, all tensors up to the second one are left orthogonalized.
            - dtol (float, optional):  The tolerance for determining non-zero eigenvalues in the SVD process done in the orthogonalization. Defaults to 1e-12.
            - normalize (bool, optional): Whether to normalize the MPS after orthogonalization. Defaults to True.

        Raises:
            - ValueError: If sites is an integer bigger than the number of sites minus one.
        """

        if sites is None:
            nstop = self.n_sites - 1
        elif isinstance(sites, int) and sites < self.n_sites:
            nstop = sites
        else:
            raise ValueError("Given sites must be an integer less than the number of sites minus one. ")

        for i in range(nstop):
            self._tensors[-i - 1], X = _right_orthogonalize_tensor(self._tensors[-i - 1], dtol)
            self._tensors[-i - 2] = ncon([self._tensors[-i - 2], X], [[-1, -2, 1], [1, -3]])

        if normalize:
            norm = ncon([self._tensors[0], np.conj(self._tensors[0])], [[1, 2, 3], [1, 2, 3]])
            self._tensors[0] /= np.sqrt(norm)

    def apply_mpo(self, mpo: "MPO"):  # sourcery skip: remove-redundant-if
        """Apply an MPO to the MPS, truncating the bond dimensions if necesary.

        Args:
            - mpo (MPO): The MPO to apply to the MPS.

        Raises:
            - ValueError: If the number of sites or physical dimension of the MPO and MPS do not match, or if the first or last tensors of the MPO are not rank-3 tensors.

            - NotImplementedError: TODO: Implement this function.
        """

        if mpo.n_sites != self.n_sites:
            raise ValueError("MPO and MPS must have the same number of sites")
        if mpo.d != self.d:
            raise ValueError("MPO and MPS must have the same physical dimension")
        if len(mpo[0].shape) != 3 or len(mpo[0].shape) != 3:
            raise ValueError("MPO first and last tensors must be a rank-3 tensors")
        if any(len(tensor.shape) != 4 for tensor in mpo[1:-1]):
            raise ValueError("Bulk tensors of the MPO must be rank-4 tensors")

        raise NotImplementedError()

    def overlap(self, other: "MPS") -> complex:
        """Compute the overlap <other|self> between two MPS.

        Args:
            other (MPS): The other MPS to compute the overlap with.

        Returns:
            complex: Value of the overlap.
        """

        if self.n_sites != other.n_sites:
            raise ValueError("MPS must have the same number of sites")

        if self.d != other.d:
            raise ValueError("MPS must have the same physical dimension")

        overlap = ncon([self._tensors[0], other._tensors[0]], [[1, 2, -1], [1, 2, -2]])
        for i in range(1, self.n_sites - 1):
            overlap = ncon([overlap, self._tensors[i], np.conj(other._tensors[i])], [[1, 2], [1, 3, -1], [2, 3, -2]])
        overlap = ncon([overlap, self._tensors[-1], np.conj(other._tensors[-1])], [[1, 2], [1, 3, 4], [2, 3, 4]])
        return overlap

    def apply_mpo(self, mpo: "MPO"):  # sourcery skip: remove-redundant-if
        """Apply an MPO to the MPS, truncating the bond dimensions if necesary.

        Args:
            - mpo (MPO): The MPO to apply to the MPS.

        Raises:
            - ValueError: If the number of sites or physical dimension of the MPO and MPS do not match, or if the first or last tensors of the MPO are not rank-3 tensors.

            - NotImplementedError: TODO: Implement this function.
        """

        if mpo.n_sites != self.n_sites:
            raise ValueError("MPO and MPS must have the same number of sites")
        if mpo.d != self.d:
            raise ValueError("MPO and MPS must have the same physical dimension")
        if len(mpo[0].shape) != 3 or len(mpo[0].shape) != 3:
            raise ValueError("MPO first and last tensors must be a rank-3 tensors")
        if any(len(tensor.shape) != 4 for tensor in mpo[1:-1]):
            raise ValueError("Bulk tensors of the MPO must be rank-4 tensors")

        raise NotImplementedError()

    def mpo_expval(self, mpo: "MPO") -> complex:
        """
        Compute the expectation value of an MPO with the MPS.

        Args:
            - mpo (MPO): The MPO to compute the expectation value with.

        Returns:
            complex: The expectation value of the MPO with the MPS.
        """

        if self.n_sites != mpo.n_sites:
            raise ValueError("MPO and MPS must have the same number of sites")
        if self.d != mpo.d:
            raise ValueError("MPO and MPS must have the same physical dimension")

        expval = ncon([self._tensors[0], mpo[0], np.conj(self._tensors[0])], [[1, 2, -1], [2, 3, -2], [1, 3, -3]])
        for i in range(1, self.n_sites - 1):
            expval = ncon(
                [expval, self._tensors[i], mpo[i], np.conj(self._tensors[i])],
                [[1, 2, 3], [1, 4, -1], [2, 4, 5, -2], [3, 5, -3]],
            )
        expval = ncon(
            [expval, self._tensors[-1], mpo[-1], np.conj(self._tensors[-1])],
            [[1, 2, 3], [1, 4, 5], [2, 4, 6], [3, 6, 5]],
        )
        return expval + mpo.offset


class MPO(TensorTrain):
    def __init__(self, n_sites, phys_d, chi, open_bounds=False):
        if open_bounds:
            raise NotImplementedError("Only closed boundary conditions are currently supported for MPO's")
        super().__init__(n_sites, phys_d, chi, open_bounds)
        self._offset = 0

    @property
    def tensors(self):
        return self._tensors

    @tensors.setter
    def tensors(self, tensors):
        # sourcery skip: remove-redundant-fstring, simplify-fstring-formatting
        # TODO These many checks may not be necessary, or they could be written in another way.

        if len(tensors[0].shape) != 3 or len(tensors[-1].shape) != 3:
            raise ValueError("Leftmost and rightmost tensors must be rank-3 tensors")

        if tensors[0].shape[0] != self.d or tensors[0].shape[1] != self.d:
            raise ValueError(
                f"Leftmost tensor's physical dimensions are {tensors[0].shape[0]} and {tensors[0].shape[1]}, but must equal physical dimension {self.d}"
            )

        if tensors[-1].shape[1] != self.d or tensors[-1].shape[2] != self.d:
            raise ValueError(
                f"Rightmost tensor's physical dimensions are {tensors[-1].shape[1]} and {tensors[-1].shape[2]}, but must equal physical dimension {self.d}"
            )

        if tensors[0].shape[2] > self.chi or tensors[-1].shape[0] > self.chi:
            raise ValueError(f"Tensor bond dimension must be less than or equal to {self.chi}")

        if tensors[0].shape[2] != tensors[1].shape[0]:
            raise ValueError(
                f"Right bond dimension of tensor {0} must be equal to left bond dimension of tensor {1} to be able to contract the MPS."
            )

        if tensors[-1].shape[0] != tensors[-2].shape[3]:
            raise ValueError(
                f"Right bond dimension of tensor {self.n_sites - 1} must be equal to left bond dimension of tensor {self.n_sites - 2} to be able to contract the MPS."
            )

        for i in range(1, len(tensors) - 1):
            if len(tensors[i].shape) != 4:
                raise ValueError("Bulk tensors must be rank-4 tensors")

            if tensors[i].shape[1] != self.d or tensors[i].shape[2] != self.d:
                raise ValueError(
                    f"Tensor's physical dimension are {tensors[i].shape[1]} and {tensors[i].shape[3]}, but must equal physical dimension {self.d}"
                )

            if tensors[i].shape[0] > self.chi or tensors[i].shape[3] > self.chi:
                raise ValueError(f"Tensor bond dimension must be less than or equal to {self.chi}")

            if tensors[i].shape[3] != tensors[i + 1].shape[0]:
                raise ValueError(
                    f"Right bond dimension of tensor {i} must be equal to left bond dimension of tensor {i + 1} to be able to contract the MPS."
                )
        self._tensors = tensors

    def from_local_ham(self, local: Hamiltonian):
        """
        Construct an MPO from a local hamiltonian.
        """
        raise NotImplementedError()

    def from_ham(self, ham: Hamiltonian):
        # sourcery skip: simplify-len-comparison, unwrap-iterable-construction
        """Generate an MPO from a general Hamiltonian with arbitrary range interactions. Terms proportional to an identity
        operator are stored in the offset attribute.

        Args:
            ham (Union[Hamiltonian, np.ndarray]): Input Hamiltonian, written in symbolic form or as a matrix.

        Raises:
            NotImplementedError: If the input Hamiltonian is not acting on an even number of qubits.
            NotImplementedError: If the hamiltonian contains interactions other than ZZ.
        """

        # TODO: Include offset inside the MPO
        n_qubits = 0
        coeffs = {}
        coeffs_x = {}
        offset = 0

        for elem in ham.parse():
            sym_names = [symbol.name for symbol in elem[1] if symbol.name != "I"]
            sym_ids = [symbol.qubit_id for symbol in elem[1] if symbol.name != "I"]
            sym_ids.sort()

            if ("Z" in sym_names and (sym_names.count("Z") != len(sym_names) or len(sym_names) > 2)
                or "X" in sym_names and (sym_names.count("X") != len(sym_names) or len(sym_names) > 1)
                or "Z" not in sym_names and "X" not in sym_names and len(sym_names) > 0):
                raise NotImplementedError("Currently only ZZ, Z and X interactions are supported")

            if "X" in sym_names:
                coeffs_x[tuple(sym_ids)] = coeffs_x.get(tuple(sym_ids), 0) + elem[0]
            else:
                coeffs[tuple(sym_ids)] = coeffs.get(tuple(sym_ids), 0) + elem[0]
            n_qubits = max(n_qubits, max(sym_ids + [0]) + 1)

            if len(sym_names) == 0:
                offset += elem[0]

        aux = False
        if n_qubits % 2 == 1:
            n_qubits += 1
            aux = True

        # Start building the Tensor Network
        i_matrix = np.array([[1, 0], [0, 1]])
        x_matrix = np.array([[0, 1], [1, 0]])
        z_matrix = np.array([[1, 0], [0, -1]])
        a_list = [
            np.zeros((min(i + 3, n_qubits - i + 2), (min(i + 3, n_qubits - i + 2)), 2, 2)) for i in range(n_qubits)
        ]

        for site in range(n_qubits):
            a_list[site][0, 0] = i_matrix
            a_list[site][-1, -1] = i_matrix
            # Set of rules
            a_list[site][0, 1] = z_matrix
            a_list[site][0, -1] = z_matrix * coeffs.get((site,), 0) + x_matrix * coeffs_x.get((site,), 0)
            for i in range(1, a_list[site].shape[0] - 2):
                a_list[site][i, i + 1] = i_matrix
            a_list[site][-2, -1] = z_matrix

        a_list[0] = a_list[0][0, :, :, :]
        a_list[-1] = a_list[-1][:, -1, :, :]

        t_list = [np.zeros((i + 3, i + 4)) for i in range(int(n_qubits / 2) - 1)]
        t_list += [np.zeros((int(n_qubits / 2) + 2, int(n_qubits / 2) + 2))]
        t_list += [np.zeros((n_qubits - i + 2, n_qubits - i + 1)) for i in range(int(n_qubits / 2), n_qubits - 1)]

        for site in range(n_qubits - 1):
            t_list[site][0, 0] = 1
            t_list[site][-1, -1] = 1
            for m in range(1, min(t_list[site].shape) - 1):
                if site + 1 < n_qubits / 2:
                    t_list[site][m, m] = 1
                    t_list[site][m, site + 2] = coeffs.get(tuple(sorted([site - m + 1, site + 1])), 0)
                elif site + 1 == n_qubits / 2:
                    for n in range(1, min(t_list[site].shape) - 1):
                        t_list[site][m, n] = coeffs.get(tuple(sorted([int(n_qubits / 2) - m, n_qubits - n])), 0)
                else:
                    t_list[site][1, m] = coeffs.get(tuple(sorted([site, n_qubits - m])), 0)
                    t_list[site][m + 1, m] = 1

        tn = [a_list[0].transpose(1, 2, 0)]
        tn += [ncon([t_list[i], a_list[i + 1]], [[-1, 1], [1, -4, -2, -3]]) for i in range(len(t_list) - 1)]
        tn += [ncon([t_list[-1], a_list[-1]], [[-1, 1], [1, -2, -3]])]

        if aux:
            tn = [i for i in tn[:-1]]
            tn[-1] = tn[-1][:, :, :, -1]

        self._tensors = tn
        self._offset = offset
        
    def from_ham_ising(self, N, J_coeffs, h_coeffs, s):
        """
        Construct an MPO for an annealing Hamiltonian for QUBO problems

        Args:
            N: number of qubits
            J_coeffs: coefficients of the long range interactions
            h_coeffs: coefficients of the short range interactions
            s: transverse field strength

        Returns:
            MPO (List[array]): List of tensors representing the MPO (1 tensor for each qubit)
        """
        # TODO Simplify MPOs if certain given coefficients are 0

        # Start building the Tensor Network
        i_matrix = np.array([[1, 0], [0, 1]])
        x_matrix = -1*np.array([[0, 1], [1, 0]])
        z_matrix = np.array([[1, 0], [0, -1]])
        
        N_by_2 = int(N / 2)

        a_list = [np.zeros((k + 2, 2,2, k + 3), dtype = complex) for k in range(1, N_by_2)]

        aux = 2 if N % 2 == 0 else 3
        a_list += [np.zeros((N_by_2 + 2, 2,2, N_by_2 + aux), dtype = complex)]

        a_list += [np.zeros((N - k + 3, 2,2, N - k + 2), dtype = complex) for k in range(N_by_2 + 1, N + 1)]

        for k in range(1, N+1):
            a_list[k-1][0,:,:,0] = i_matrix
            a_list[k-1][-1,:,:,-1] = i_matrix
            a_list[k-1][-2,:,:,-1] = z_matrix
            a_list[k-1][0,:,:,-1] = (-1/N)*(1-s)*x_matrix + s*h_coeffs[k-1]*z_matrix
            
            if k < N_by_2:
                a_list[k-1][0,:,:,1] = z_matrix 
                a_list[k-1][0,:,:,k+1] = s*J_coeffs[k-1,k]*z_matrix

                for m in range(2, k+1):
                    a_list[k-1][m-1,:,:,m] = i_matrix
                    a_list[k-1][m-1,:,:,k+1] = s*J_coeffs[k-m, k]*i_matrix

            elif k == N_by_2:
                for n in range(2, N_by_2 + aux):
                    a_list[k-1][0,:,:,n-1] = s*J_coeffs[N_by_2-1, N-n+1]*z_matrix
                    for m in range(2, k+1):
                        a_list[k-1][m-1,:,:,n-1] = s*J_coeffs[N_by_2-m,N-n+1]*i_matrix

            else: # k > N_by_2:
                for m in range(2, N-k+2):
                    a_list[k-1][0,:,:,m-1] = s*J_coeffs[k-1,N-m+1]*z_matrix
                    a_list[k-1][m-1,:,:,m-1] = i_matrix

        a_list[0] = a_list[0][0, :] # other rows are used to propagate information of previous tensors, does not make sense to keep them
        a_list[-1] = a_list[-1][:, -1] # other columns are used to propagate information to the next tensors, does not make sense to keep them

        self._tensors = a_list
        
        #return a_list

    @property
    def offset(self):
        """
        Returns the energy offset value of the Hamiltonian.

        Returns:
            float: The offset value.
        """
        return self._offset


class MPSGenerator:
    """
    Class representing different types of MPS.

    Args:
        - d (int): physical index dimension of the MPS.

        - chi (int): maximum permitted internal dimensions.

        - n_sites (int): number of tensioners making up the MPS.

    Methods:
        - random_complex_mps() -> np.ndarray:
            Constructs an complex MPS whose internal values are randomised.

        - random_real_mps() -> np.ndarray:
            Constructs an real MPS whose internal values are randomised.

        - non_mps() -> np.ndarray:
            Constructs an MPS whose internal values are zeros.

        - zeros_mps() -> np.ndarray:
            Constructs an MPS that represents the product state zeros: |0000...>.
    """

    def __init__(self, d: int, chi: int, n_sites: int):
        self.d = d
        self.chi = chi
        self.n_sites = n_sites

    def random_complex_mps(self) -> np.array:
        """
         Constructs an complex MPS whose internal values are randomised.

        Returns:
            - np.ndarray: the np.array contains in sorted form the tensors that make up the random MPS
        """
        tensors = [0 for _ in range(self.n_sites)]

        tensors[0] = np.random.rand(1, self.d, min(self.d, self.chi)) + 1j * np.random.rand(
            1, self.d, min(self.d, self.chi)
        )

        for i in range(1, self.n_sites):

            tensors[i] = np.random.rand(
                tensors[i - 1].shape[2],
                self.d,
                min(self.chi, tensors[i - 1].shape[2] * self.d, self.d ** (self.n_sites - i - 1)),
            ) + 1j * np.random.rand(
                tensors[i - 1].shape[2],
                self.d,
                min(self.chi, tensors[i - 1].shape[2] * self.d, self.d ** (self.n_sites - i - 1)),
            )

        return tensors

    def random_real_mps(self) -> np.array:
        """
            Constructs an real MPS whose internal values are randomised.

        Returns:
            - np.ndarray: the np.array contains in sorted form the tensors that make up the random MPS
        """
        tensors = [0 for _ in range(self.n_sites)]

        tensors[0] = np.random.rand(1, self.d, min(self.d, self.chi)) 

        for i in range(1, self.n_sites):

            tensors[i] = np.random.rand(
                tensors[i - 1].shape[2],
                self.d,
                min(self.chi, tensors[i - 1].shape[2] * self.d, self.d ** (self.n_sites - i - 1)),
            )

        return tensors

    def non_mps(self) -> np.array:
        """
         Constructs an MPS whose internal values are zeros.

        Returns:
            - np.ndarray: the np.array contains in sorted form the tensors that make up the non_mps.
        """

        tensors = [0 for _ in range(self.n_sites)]

        tensors[0] = np.zeros((1, self.d, min(self.d, self.chi)))

        for i in range(1, self.n_sites):

            tensors[i] = np.zeros(
                (
                    tensors[i - 1].shape[2],
                    self.d,
                    min(self.chi, tensors[i - 1].shape[2] * self.d, self.d ** (self.n_sites - i - 1)),
                )
            )

        return tensors

    def zeros_mps(self) -> np.array:
        """
         Construct an MPS that represents the product state zeros: |0000...>.

        Returns:
            - np.ndarray: the np.array contains in sorted form the tensors that make up the zeros_mps.
        """

        tensors = self.non_mps()

        for j in range(self.n_sites):

            tensors[j][0][0][0] = 1

        return tensors


class TensorOperations:
    """
    Class representing different operations that can be performed on the class MPSGenerator.

    Args:
        - mps_list (MPSGenerator): np.array containing an MPS of class MPSGenerator.

    Internal Methods:

        - r_tensor() -> (np.array, np.array):
            Calculate the X and Xinv matrices to the right of the specified link.

        - l_tensor() -> (np.array, np.array):
            Calculate the X and Xinv matrices to the left of the specified link.

        - left_orthogonal_single_tensor() -> np.array:
            Converts an MPS tensor into a left form tensor.

    Methods:

        - proyection_state() -> float:
            Projects the MPS state onto a specific state indicated by a str representing the state to be projected onto.
            |<MPS|0000>| == TensorOperations(mps_list = MPS).proyection_state(s="0" * 4)

        - canonical_network() -> list(np.array)
            Calculate the canonical form of the MPS.

        - contraction_network() -> np.array:
            Calculates the contraction of an mps as a single tensor.

        - contraction_canonical_network() -> np.array:
            Calculates the contraction of an mps as a single tensor. The MPS state must be in its canonical form.

        - mps_canonical_to_mps() -> list(np.array()):
            Transforms the MPS in canonical form to a standard MPS.

        - left_orthogonal_form() -> list(np.array()):
            Converts the MPS into a left form.

        - truncation_group_network() -> list(np.array()):
            Truncates the MPS state to another MPS of lower chi using cluster truncation.

        - truncation_network_left_form() -> list(np.array()):
            Truncate the MPS state to another MPS of lower chi using left form truncation.

        - truncation_canonical_network() -> list(np.array()):
            Truncate the MPS state to another MPS of lower chi using canonical truncation.
            The MPS must be in its canonical form.
    """

    def __init__(self, mps_list: np.array):

        self.mps_list = mps_list

    def proyection_state(self, s: str) -> float:
        """
         Projects the MPS state onto a specific state indicated by a str representing the state to be projected onto.
        |<MPS|0000>| == proyection_state(s = '0000')

        Args:
            - s (str): string representing the state on which the projection is to be made.

        Returns:
            - float: the float represents the overlap between the MPS state and the target state.
        """

        n_sites = len(self.mps_list)

        if isinstance(s, str):

            s = [int(char) for char in s]
        else:

            raise ValueError("Incorrect format")

        if n_sites == len(s):

            p = self.mps_list[0][:, s[0], :]

            for i in range(1, n_sites):
                p = p @ self.mps_list[i][:, s[i], :]

            return p[0][0]

        else:

            raise ValueError("Error: the MPS and the bit string have different lengths.")

    def proyection_mps_operator(self, mps_bra: np.array) -> float:
        """_summary_

        Args:
            mps_bra (_type_): _description_
            mps_ket (_type_): _description_

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """

        if len(mps_bra) != len(self.mps_list):

            raise ValueError("The length of the MPS are different")

        tensor = np.eye(1)

        for i in range(len(mps_bra) - 1):

            tensor = ncon([tensor, self.mps_list[i], np.conj(mps_bra[i])], [[1, 2], [1, 3, -1], [2, 3, -2]])

        tensor = ncon([tensor, self.mps_list[-1], np.conj(mps_bra[-1])], [[1, 2], [1, 3, 4], [2, 3, 4]])

        return tensor
    
    
    def _r_tensor(self, right_enviroment: np.array, dtol: Optional[float] = 1e-12) -> (np.array, np.array):  # type: ignore
        """
        Calculate the X and Xinv matrices to the right of the specified link.

        Args:
            - right_tensors (MPSGenerator): MPS tensioners on the right hand side of the link.

        Returns:
            - (np.array, np.array): X, X_inv.
        """

        d, u = LA.eigh(right_enviroment)

        n_values = sum(d > dtol)

        d_v = d[range(-1, -n_values - 1, -1)]
        u_m = u[:, range(-1, -n_values - 1, -1)]

        sq_d_v = np.sqrt(abs(d_v))

        x = u_m @ np.diag(sq_d_v) @ np.conj(u_m.T)
        x_inv = u_m @ np.diag(1 / sq_d_v) @ np.conj(u_m.T)

        return x, x_inv

    def _l_tensor(self, left_tensors: np.array, dtol: Optional[float] = 1e-12) -> (np.array, np.array):  # type: ignore
        """
        Calculate the X and Xinv matrices to the left of the specified link.

        Args:
            - left_tensors (MPSGenerator): MPS tensioners on the left hand side of the link.

        Returns:
            - (np.array, np.array): X, X_inv.
        """

        if len(left_tensors)==1:
            
            left_enviroment = ncon([left_tensors[-1], np.conj(left_tensors[-1])],[[1, 2, -1], [1, 2, -2]])
        
        else:
            
            left_enviroment = ncon([left_tensors[-2]@np.conj(left_tensors[-2]), left_tensors[-1], np.conj(left_tensors[-1])],[[1, 2], [1, 3, -1], [2, 3, -2]])
        
        d, u = LA.eigh(left_enviroment)

        n_values = sum(d > dtol)

        d_v = d[range(-1, -n_values - 1, -1)]
        u_m = u[:, range(-1, -n_values - 1, -1)]

        sq_d_v = np.sqrt(abs(d_v))

        x = u_m @ np.diag(sq_d_v) @ np.conj(u_m.T)
        x_inv = u_m @ np.diag(1 / sq_d_v) @ np.conj(u_m.T)

        return np.conj(x), np.conj(x_inv)
    
    def _r_environment_tensors(self, tensors: np.array) -> list:
        """
        Calculates the environment of the tensor set to the right of the link to be used iteratively

        Args:
            - tensors (np.array): all mps tensors.

        Returns:
            - list(np.array): list of density matrices resulting 
                from the contraction of the MPS with its conjugate
        """
        
        list_env_r_tensors = []
        
        tensors_r = list(reversed(tensors))

        rho_r_v = ncon([tensors_r[0], np.conj(tensors_r[0])], [[-1, 1, 2], [-2, 1, 2]])
        
        list_env_r_tensors.append(rho_r_v)

        if len(tensors_r) > 1:

            for i in range(1, len(tensors_r) - 1):

                rho_r_v = ncon(
                    [tensors_r[i], np.conj(tensors_r[i]), rho_r_v], [[-1, 1, 2], [-2, 1, 3], [2, 3]]
                )
                
                list_env_r_tensors.append(rho_r_v)
        
        list_env_r_tensors = list(reversed(list_env_r_tensors))
                
        return list_env_r_tensors


    def canonical_network(self, chi: Optional[int] = None) -> np.array:
        """
        Calculate the canonical form of the MPS. In the canonical form,
        the new MPS has interleaved modified tensors and singular value matrices.
        
        Args:
            - chi (int): the maximum value of internal dimension.

        Returns:
            - list(np.array): contains a new MPS in its canonical form where the elements
                in the odd positions represent the singular value matrices
        """
        
        if not chi:
            chi = int(self.mps_list[int(len(self.mps_list)//2)].shape[2])*2
        
        n_sites = len(self.mps_list)

        tensor_list = self.mps_list.copy()
        
        list_env_r_tensors = self._r_environment_tensors(tensors=self.mps_list.copy())
        
        canonical_tensors = [tensor_list[0]]

        for i in range(1, n_sites):

            x_l, x_inv_l = self._l_tensor(left_tensors = canonical_tensors)
            x_r, x_inv_r = self._r_tensor(right_enviroment = list_env_r_tensors[0])
            
            list_env_r_tensors.pop(0)
            
            tensor_l = ncon([canonical_tensors[-1], x_inv_l], [[-1, -2, 1], [1, -3]])
            tensor_r = ncon([x_inv_r, tensor_list[i]], [[-1, 1], [1, -2, -3]])

            utemp, sig_p, vhtemp = LA.svd(x_l @ x_r)
            
            tensor_l_p = ncon([tensor_l, utemp], [[-1, -2, 1], [1, -3]])
            tensor_r_p = ncon([vhtemp, tensor_r], [[-1, 1], [1, -2, -3]])
            

            canonical_tensors[-1] = tensor_l_p[:,:,:chi]
            canonical_tensors.append(np.diag(sig_p[:chi]))
            canonical_tensors.append(tensor_r_p[:chi,:,:])
            
        return canonical_tensors

    def contraction_network(self) -> np.array:
        """
        Calculates the contraction of an mps as a single tensor.

        Returns:
            - np.array: represents the fully contracted MPS state.
        """

        n_sites = len(self.mps_list)

        indice_list = [[-1, -2, 1]]

        for i in range(1, n_sites):

            lista = [i, -(i + 2), -(i + 3)] if i == n_sites - 1 else [i, -(i + 2), i + 1]
            indice_list.append(lista)

        return ncon(self.mps_list, indice_list)

    def contraction_canonical_network(self) -> np.array:
        """
        Calculates the contraction of an mps as a single tensor. The MPS state must be in its canonical form.

        Returns:
            - np.array: represents the fully contracted MPS state.
        """

        n_sites = len(self.mps_list)

        indice_list = [[-1, -2, 1]]

        for i in range(1, n_sites):

            if i % 2 == 0:

                lista = [i, -(i + 1), -(i + 2)] if i == n_sites - 1 else [i, -(i + 1), i + 1]
            else:
                lista = [i, i + 1]

            indice_list.append(lista)

        return ncon(self.mps_list, indice_list)

    def mps_canonical_to_mps(self) -> np.array:
        """
        Transforms the MPS in canonical form to a standard MPS.

        Returns:
            - np.array: represents the MPS state in canonical form in an MPS state.
        """

        tensor_list_mps = [
            ncon([self.mps_list[i], self.mps_list[i + 1]], [[-1, -2, 1], [1, -3]])
            for i in range(0, len(self.mps_list) - 1, 2)
        ]
        tensor_list_mps.append(self.mps_list[-1])

        return tensor_list_mps

    def _left_orthogonal_single_tensor(self, tensor: np.array, dtol: Optional[float] = 1e-12) -> (np.array, np.array):  # type: ignore
        """
        Converts an MPS tensor into a left form tensor.

        Args:
            - tensor (int): the tensor to convert in to a left form.
            - dtol (float): threshold to eliminate eigenvalues

        Returns:
            - np.array: represents the tensor in left form.
        """

        rho = ncon([tensor, np.conj(tensor)], [[1, 2, -1], [1, 2, -2]])

        dtemp, utemp = LA.eigh(rho)

        chitemp = sum(dtemp > dtol)

        d = dtemp[range(-1, -chitemp - 1, -1)]
        u = utemp[:, range(-1, -chitemp - 1, -1)]

        sq_d = np.sqrt(abs(d))
        x = np.conj(u) @ np.diag(sq_d) @ u.T
        x_inv = np.conj(u) @ np.diag(1 / sq_d) @ u.T

        new_tensor = ncon([tensor, x_inv], [[-1, -2, 1], [1, -3]])

        return new_tensor, x

    def left_orthogonal_form(self, nstop: Optional[int] = 0, dtol: Optional[float] = 1e-12) -> np.array:
        """
        Converts the MPS into a left form.

        Args:
            - nstop (int): number of MPS tensors to convert to left form, if the value is zero
              the whole MPS is converted to left form.
            - dtol (float): threshold to eliminate eigenvalues

        Returns:
            - np.array: represents the MPS in left form.
        """

        tensor_list = self.mps_list.copy()

        if nstop == 0:
            nstop = len(tensor_list) - 1

        for i in range(nstop):
            tensor_list[i], x = self._left_orthogonal_single_tensor(tensor_list[i], dtol)
            tensor_list[i + 1] = ncon([x, tensor_list[i + 1]], [[-1, 1], [1, -2, -3]])

        return tensor_list

    def truncation_group_network(self, chi: int) -> np.array:
        """
        Truncates the MPS state to another MPS of lower chi using cluster truncation.

        Args:
            - chi (int): maximum dimension at which to truncate the MPS

        Returns:
            - np.array: represents the truncate MPS state.
        """

        tensor_list = self.mps_list.copy()

        tensor_truncation_list = []

        for i in range(len(tensor_list) - 1):

            tensor_c = ncon([tensor_list[i], tensor_list[i + 1]], [[-1, -2, 1], [1, -3, -4]])

            left = tensor_list[i].shape[0] * tensor_list[i].shape[1]
            right = tensor_list[i + 1].shape[1] * tensor_list[i + 1].shape[2]

            tensor_c_svd = tensor_c.reshape(left, right)

            u_m, s_m, v_h = LA.svd(tensor_c_svd, full_matrices=False)

            u_m = u_m.reshape(tensor_list[i].shape[0], tensor_list[i].shape[1], s_m.shape[0])
            v_h = v_h.reshape(s_m.shape[0], tensor_list[i + 1].shape[1], tensor_list[i + 1].shape[2])

            u_t = u_m[:, :, : min(chi, tensor_list[i].shape[2])]
            s_t = np.diag(s_m[: min(chi, tensor_list[i].shape[2])])
            v_t = v_h[: min(chi, tensor_list[i].shape[2]), :, :]

            tensor_c_final = ncon([u_t, s_t], [[-1, -2, 1], [1, -3]])

            tensor_truncation_list.append(tensor_c_final)

            tensor_list[i + 1] = v_t

        tensor_truncation_list.append(tensor_list[-1])

        return tensor_truncation_list

    def truncation_network_left_form(self, chi: Union[int, list]) -> np.array:
        """
        Truncate the MPS state to another MPS of lower chi using left form truncation.

        Args:
            - chi (Union[int, list]): maximum dimension at which to truncate the MPS

        Returns:
            - np.array: represents the truncate MPS state.
        """

        tensor_list = self.mps_list.copy()

        if isinstance(chi, int):
            chi = [chi] * (len(tensor_list) - 1)
        if not isinstance(chi, int) and not isinstance(chi, list):
            raise ValueError("The variable chi expects an integer value or in list format")

        for i in range(len(tensor_list) - 1):

            tensor_c_svd = tensor_list[i].reshape(
                tensor_list[i].shape[0] * tensor_list[i].shape[1], tensor_list[i].shape[2]
            )

            u_m, s_m, v_h = LA.svd(tensor_c_svd, full_matrices=False)

            u_m = u_m.reshape(tensor_list[i].shape[0], tensor_list[i].shape[1], s_m.shape[0])

            v_ht = v_h[: chi[i], :]
            s_mt = np.diag(s_m[: chi[i]])
            u_mt = u_m[:, :, : chi[i]]

            tensor_list[i] = u_mt

            u_l = ncon([s_mt, v_ht, tensor_list[i + 1]], [[-1, 1], [1, 2], [2, -2, -3]])

            tensor_list[i + 1] = u_l

        return tensor_list

    def truncation_canonical_network(self, chi: Union[int, list]) -> np.array:
        """
        Truncate the MPS state to another MPS of lower chi using canonical truncation.
        The MPS must be in its canonical form.

        Args:
            - chi (Union[int, list]): maximum dimension at which to truncate the MPS

        Returns:
            - np.array: represents the truncate MPS state in canonical form.
        """
        tensor_list = self.mps_list.copy()

        if isinstance(chi, int):
            chi = [chi] * (len(tensor_list) - 1)
        if not isinstance(chi, int) and not isinstance(chi, list):
            raise ValueError("The variable chi expects an integer value or in list format")

        tensor_truncation_list = [tensor_list[0][:, :, : chi[0]]]

        for i in range(1, len(tensor_list) - 1):

            if i % 2 == 0:

                tensor_truncation_list.append(tensor_list[i][: chi[i], :, : chi[i]])
            else:

                tensor_truncation_list.append(tensor_list[i][: chi[i], : chi[i]])

        tensor_truncation_list.append(tensor_list[-1][: chi[-1], :, :])

        return tensor_truncation_list


class TensorGates:
    """_summary_"""

    def __init__(self): ...

    def x(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return np.array([[0, 1], [1, 0]])

    def y(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return np.array([[0, -1j], [1j, 0]])

    def z(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return np.array([[1, 0], [0, -1]])

    def h(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return np.array([[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]])

    def cz(self, split=True):  # sourcery skip: class-extract-method
        """_summary_

        Args:
            split (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        cz = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])

        if split:

            u_m, s_m, v_h = LA.svd(cz, full_matrices=False)

            u_l = u_m.reshape(2, 2, 4)

            u_r = ncon([np.diag(s_m), v_h], [[-1, 1], [1, -2]])

            u_r = u_r.reshape(4, 2, 2)

            return u_l, u_r

        return cz.reshape((2, 2, 2, 2))

    def cnot(self, split=True):
        """_summary_

        Args:
            split (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """

        cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

        if split:

            u_m, s_m, v_h = LA.svd(cnot, full_matrices=False)

            u_l = u_m.reshape(2, 2, 4)

            u_r = ncon([np.diag(s_m), v_h], [[-1, 1], [1, -2]])

            u_r = u_r.reshape(4, 2, 2)

            return u_l, u_r

        return cnot.reshape((2, 2, 2, 2))

    def rx(self, angle):
        """_summary_

        Args:
            angle (_type_): _description_

        Returns:
            _type_: _description_
        """

        return np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)], [-1j * np.sin(angle / 2), np.cos(angle / 2)]])

    def ry(self, angle):
        """_summary_

        Args:
            angle (_type_): _description_

        Returns:
            _type_: _description_
        """

        return np.array([[np.cos(angle / 2), -np.sin(angle / 2)], [np.sin(angle / 2), np.cos(angle / 2)]])

    def rz(self, angle):
        """_summary_

        Args:
            angle (_type_): _description_

        Returns:
            _type_: _description_
        """

        return np.array(
            [[np.cos(angle / 2) + 1j * np.sin(angle / 2), 0], [0, np.cos(angle / 2) - 1j * np.sin(angle / 2)]]
        )
