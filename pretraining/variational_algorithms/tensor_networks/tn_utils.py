from typing import List


class StructureResultsPretraining:
    def __init__(
        self,
        effective_dimension: List,
        coherence: List,
        fun: float,
        fidelity: float,
        fidelity_history: List,
        fun_history: List,
        circuit_mps,
        circuit_su4=None,
    ):

        self.fun = fun
        self.effective_dimension = effective_dimension
        self.coherence = coherence
        self.fun_history = list(fun_history)
        self.fidelity = fidelity
        self.fidelity_history = list(fidelity_history)
        self.circuit_mps = circuit_mps
        self.circuit_su4 = circuit_su4

    def __str__(self):
        """Structure the results obtained in the FVQE algorithm so that it has the same form as the rest of the
        optimizers.

        Args:
            - self.fun_dmrg: indicates the energy obtained by the algorithm dmrg.
            - self.fun_dmrg_history: indicates the energy history obtained by the dmrg algorithm at each optimization step.
            - self.fidelity: indicates the fidelity obtained by the subroutine mps to pqc.
            - self.fidelity_history: indicates the fidelity history obtained through the mps to pqc subroutine.
            - self.circuit_mps: indicates the object containing the circuit representing the final state mps.
            - self.circuit_su4: indicates the final layer containing the su(4) gates in full connected format.

        Returns:

            - returns an object containing all structured args.
        """

        return f"Results:\neffective_dimension: {self.effective_dimension}\ncoherence: {self.coherence} \nfun: {self.fun} \nfun_history: {self.fun_history}\nfidelity: {self.fidelity}\nfidelity_history: {self.fidelity_history}\ncircuit_mps: {self.circuit_mps}\ncircuit_su4: {self.circuit_su4}"
