import copy
import enum
from typing import Dict, List, Tuple, Union

import numpy as np

### Utils ###

def _multiply_pauli(op1: "PauliOperator", op2: "PauliOperator"):
    if op1.qubit_id != op2.qubit_id:
        raise ValueError("Operators must act on the same qubit")
    if op1.name == "X":
        if op2.name == "X":
            return 1, I(0)
        if op2.name == "Y":
            return 1j, Z(op1.qubit_id)
        if op2.name == "Z":
            return -1j, Y(op1.qubit_id)
        if op2.name == "I":
            return 1, op1
    if op1.name == "Y":
        if op2.name == "X":
            return -1j, Z(op1.qubit_id)
        if op2.name == "Y":
            return 1, I(0)
        if op2.name == "Z":
            return 1j, X(op1.qubit_id)
        if op2.name == "I":
            return 1, op1
    if op1.name == "Z":
        if op2.name == "X":
            return 1j, Y(op1.qubit_id)
        if op2.name == "Y":
            return -1j, X(op1.qubit_id)
        if op2.name == "Z":
            return 1, I(0)
        if op2.name == "I":
            return 1, op1
    if op1.name == "I":
        if op2.name == "X":
            return 1, X(op1.qubit_id)
        if op2.name == "Y":
            return 1, Y(op1.qubit_id)
        if op2.name == "Z":
            return 1, Z(op1.qubit_id)
        if op2.name == "I":
            return 1, I(0)
    raise NotImplementedError(f"Operation between operator {op1.name} and operator {op2.name} is not supported.")


def A(qubit_id):
    """A method that returns the operator with eigenvalues 0 and 1 as a function of Pauli Z"""
    return (1 - Z(qubit_id)) / 2


def multiply_sets(set1: Tuple["PauliOperator"], set2: Tuple["PauliOperator"]) -> Tuple[Union[int, float, complex], "PauliOperator"]:

    list1 = list(set1)
    list2 = list(set2) 
    
    list1.extend(list2) 

    combined_list = sorted(list1, key=lambda x: x[0])   

    sum_dict = {}

    for qid, op in combined_list:
        if  qid not in sum_dict.keys():
            sum_dict[qid] = []
        sum_dict[qid].append(pauli_map[op](qid))

    simplified_list = []
    accumulated_phase = 1
    for v in sum_dict.values():
        op1 = v[0]
        phase = 1
        if len(v) > 1:
            for i in range(1, len(v)):
                aux_phase, op1 = _multiply_pauli(op1, v[i])
                phase *= aux_phase
                
        simplified_list.append((op1.qubit_id, op1.name))
        accumulated_phase *= phase 
        
    return accumulated_phase, tuple(simplified_list)
        

class Side(enum.Enum):
    RIGHT = "right"
    LEFT = "left"

### Operations ###


class Operation(enum.Enum):
    MUL = "*"
    ADD = "+"
    DIV = "/"
    SUB = "-"


class ComparisonOperators(enum.Enum):
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="


### Pauli ###


class PauliOperator():
    """Abstract Representation of a generic Pauli operator

    Args:
        - qubit_id (int): the qubit that the operator will be acting on
        - name (str): the name of the Pauli operator
        - matrix : the matrix representation of the Pauli operator

    Attributes:
        - qubit_id (int): the qubit that the operator will be acting on
        - name (str): the name of the Pauli operator
        - matrix : the matrix representation of the Pauli operator

    Methods:
        - parse: yields the operator in the following format
                    (1, [<operator>])
    """

    def __init__(self, qubit_id: int, name: str, matrix=None) -> None:
        self.qubit_id = qubit_id
        self.__matrix = matrix
        self._name = name
        self.coeff = 1 + 0j 

    @property
    def name(self) -> str:
        """The name of the Pauli operator.

        Returns:
            str: the name of the pauli operator.
        """
        return self._name

    @property
    def matrix(self):
        """The matrix of the Pauli operator.

        Returns:

        """
        return self.__matrix

    def parse(self):
        yield 1, [self]

    def __copy__(self):
        return PauliOperator(name=self.name, qubit_id=self.qubit_id, matrix=self.matrix)

    def __repr__(self):
        return f"{self.name}({self.qubit_id})"

    def __str__(self):
        return f"{self.name}({self.qubit_id})"

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out + other
    
    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out + other
    
    def __iadd__(self, other):
        
        if isinstance(other, (int, float)) and other == 0:
            return self
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out + other

    def __sub__(self, other):
        
        if isinstance(other, PauliOperator) and other.name == self.name and other.qubit_id == self.qubit_id:
            return 0
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out - other
    
    def __rsub__(self, other):
        
        if isinstance(other, PauliOperator) and other.name == self.name and other.qubit_id == self.qubit_id:
            return 0
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return  other - out
    
    def __isub__(self, other):
        
        if isinstance(other, PauliOperator) and other.name == self.name and other.qubit_id == self.qubit_id:
            return 0
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out - other

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return 0
            elif other == 1:
                return self
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out * other

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return 0
            elif other == 1:
                return self
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return other * out
    
    def __imul__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                return 0
            elif other == 1:
                return self
        
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out * other

    def __truediv__(self, other):
        out = Hamiltonian({((self.qubit_id, self.name),): 1})
        
        return out / other 
    
    def __rtruediv__(self, other):
        raise ValueError("Division by operators is not supported")
    
    def __itruediv__(self, other):
        out = Hamiltonian({((self.qubit_id, self.name),): 1})    
        return out / other 


class Z(PauliOperator):
    """The Pauli Z operator"""

    def __init__(self, qubit_id: int) -> None:
        """constructs a new Pauli Z operator

        Args:
            qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[1, 0], [0, -1]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return Z(qubit_id=self.qubit_id)


class X(PauliOperator):
    """The Pauli X operator"""

    def __init__(self, qubit_id: int) -> None:
        """Constructs a new Pauli X operator

        Args:
            - qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[0, 1], [1, 0]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return X(qubit_id=self.qubit_id)


class Y(PauliOperator):
    """The Pauli Y operator"""

    def __init__(self, qubit_id: int) -> None:
        """Constructs a new Pauli Y operator

        Args:
            qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[0, 1j], [1j, 0]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return Y(qubit_id=self.qubit_id)


class I(PauliOperator):
    """The Identity operator"""

    def __init__(self, qubit_id: int) -> None:
        """Create a new Identity operator

        Args:
            qubit_id (int): the qubit that the operator will act on.
        """
        matrix = [[1, 0], [0, 1]]
        name = self.__class__.__name__
        super().__init__(qubit_id=qubit_id, name=name, matrix=matrix)

    def __copy__(self):
        return I(qubit_id=self.qubit_id)



pauli_map = {"Z": Z, "X" : X, "Y" : Y, "I" : I}




### Terms ###



class Hamiltonian:
    '''
    Assumes elements have the following encoding: 
        {
            ((qubit_id, pauli_operator), ...) : coefficient,
        }
        
        example:
        {
            ((0, Z(0)), (1, Y(1))): 1,
            ((1, X(1)),): 1j,
        }
    '''
    def __init__(self, elements: Dict[Tuple[Tuple[int, PauliOperator]], Union[int, float, complex]]) -> None:
        self.__elements = {}
        for operators, coeff in elements.items():
            self.__elements[operators] = coeff 
        
    @property
    def elements(self):
        return self.__elements

    def variables(self):
        for key in self.elements.keys:
            for qid, operator in key:
                yield pauli_map[operator](qid)

    def parse(self):
        for key, value in self.elements.items():
            yield value, [pauli_map[op](qid) for qid, op in key]
        
    def filter(self):
        pop_keys = []
        for key, value in self.elements.items():
            if np.real_if_close(value) == 0:
                pop_keys.append(key)
        
        for key in pop_keys:
            self.elements.pop(key)

    def __copy__(self):
        return Hamiltonian(elements=self.elements.copy())
    
    def __repr__(self):
        out = ""
        for operators, coeff in self.elements.items():
            coeff = np.real_if_close(coeff)
            if out == "": 
                if isinstance(coeff, (complex)) or np.iscomplex(coeff):
                    out += f"({coeff}) "
                else: 
                    if coeff != 1:
                        if coeff == -1:
                            out += "-"
                        out += f"{coeff} "
            else: 
                if isinstance(coeff, (complex)) or np.iscomplex(coeff):
                    out += f"+ ({coeff}) "
                else: 
                    if coeff != 1 and coeff != -1:
                        out += f"+ {coeff} " if coeff > 0 else f"- {np.abs(coeff)} "
                    else: 
                        out += f"+ " if coeff > 0 else f"- "
                    
            for qid, o in operators:
                out += f"{o}({qid}) "
        return out
                
    def __str__(self):
        out = ""
        for operators, coeff in self.elements.items():
            coeff = np.real_if_close(coeff)
            if out == "": 
                if isinstance(coeff, (complex)) or np.iscomplex(coeff):
                    out += f"({coeff}) "
                else: 
                    if coeff != 1:
                        if coeff == -1:
                            out += "-"
                        out += f"{coeff} "
            else: 
                if isinstance(coeff, (complex)) or np.iscomplex(coeff):
                    out += f"+ ({coeff}) "
                else: 
                    if coeff != 1 and coeff != -1:
                        out += f"+ {coeff} " if coeff > 0 else f"- {np.abs(coeff)} "
                    else: 
                        out += f"+ " if coeff > 0 else f"- "
                    
            for qid, o in operators:
                out += f"{o}({qid}) "
        return out

    def __getitem__(self,index):
         return self.elements[index]
     
    def __setitem__(self, key, value):
         self.elements[key] = value
    
    def __add__(self, other):
        out = copy.copy(self) 

        if isinstance(other, Hamiltonian):
            for key, value in other.elements.items():
                if key in out.elements.keys():
                    out[key] += value
                else: 
                    out[key] = value 
        elif isinstance(other, PauliOperator):
            encoded = ((other.qubit_id, other.name), )
            if encoded in out.elements.keys():
                out[encoded] += 1
            else:
                out[encoded] = 1 
        elif isinstance(other, (int, float, complex)):
            if ((0, "I"),) in out.elements.keys():
                out[((0, "I"),)] += other 
            else:
                out[((0, "I"),)] = other 
        else:
            raise ValueError(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.filter()
        return out
       
    def __mul__(self, other):
        out = Hamiltonian({})

        if isinstance(other, Hamiltonian):
            # unfold parenthesis
            for key1 in self.elements.keys():
                for key2 in other.elements.keys():
                    phase, new_key = multiply_sets(key1, key2)
                    if new_key in out.elements.keys():
                        out[new_key] += phase * self[key1] * other[key2]
                    else: 
                        out[new_key] = phase * self[key1] * other[key2]
        elif isinstance(other, PauliOperator):
            key2 = ((other.qubit_id, other.name), )
            for key1 in self.elements.keys():
                phase, new_key = multiply_sets(key1, key2)
                if new_key in out.elements.keys():
                    out[new_key] += phase * self[key1]
                else: 
                    out[new_key] = phase * self[key1]
        elif isinstance(other, (int, float, complex)):
            out = copy.copy(self) 
            for key in out.elements.keys():
                out[key] *= other
            return out
        else:
            raise ValueError(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.filter()
        return out

    def __truediv__(self, other):
        if not isinstance(other, (float, int, complex)):
            raise ValueError("Division of operators is not supported")
       
        return self * (1/other) 

    def __sub__(self, other):
        return self + (-1 * other)

    def __iadd__(self, other):
        out = copy.copy(self) 

        if isinstance(other, Hamiltonian):
            for key, value in other.elements.items():
                if key in out.elements.keys():
                    out[key] += value
                else: 
                    out[key] = value 
        elif isinstance(other, PauliOperator):
            encoded = ((other.qubit_id, other.name), )
            if encoded in out.elements.keys():
                out[encoded] += 1
            else:
                out[encoded] = 1 
        elif isinstance(other, (int, float, complex)):
            if ((0, "I"),) in out.elements.keys():
                out[((0, "I"),)] += other 
            else:
                out[((0, "I"),)] = other 
        else:
            raise ValueError(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.filter()
        return out

    def __imul__(self, other):
        out = Hamiltonian({})

        if isinstance(other, Hamiltonian):
            # unfold parenthesis
            for key1 in self.elements.keys():
                for key2 in other.elements.keys():
                    phase, new_key = multiply_sets(key1, key2)
                    if new_key in out.elements.keys():
                        out[new_key] += phase * self[key1] * other[key2]
                    else: 
                        out[new_key] = phase * self[key1] * other[key2]
        elif isinstance(other, PauliOperator):
            key2 = ((other.qubit_id, other.name), )
            for key1 in self.elements.keys():
                phase, new_key = multiply_sets(key1, key2)
                if new_key in out.elements.keys():
                    out[new_key] += phase * self[key1]
                else: 
                    out[new_key] = phase * self[key1]
        elif isinstance(other, (int, float, complex)):
            out = copy.copy(self) 
            for key in out.elements.keys():
                out[key] *= other
            return out
        else:
            raise ValueError(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.filter()
        return out

    def __itruediv__(self, other):
        if not isinstance(other, (float, int, complex)):
            raise ValueError("Division of operators is not supported")
       
        return self * (1/other) 

    def __isub__(self, other):
        return self + (-1 * other)
    
    def __radd__(self, other):
        out = copy.copy(self) 

        if isinstance(other, Hamiltonian):
            for key, value in other.elements.items():
                if key in out.elements.keys():
                    out[key] += value
                else: 
                    out[key] = value 
        elif isinstance(other, PauliOperator):
            encoded = ((other.qubit_id, other.name), )
            if encoded in out.elements.keys():
                out[encoded] += 1
            else:
                out[encoded] = 1 
        elif isinstance(other, (int, float, complex)):
            if ((0, "I"),) in out.elements.keys():
                out[((0, "I"),)] += other 
            else:
                out[((0, "I"),)] = other 
        else:
            raise ValueError(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.filter()
        return out

    def __rmul__(self, other):
        out = Hamiltonian({})

        if isinstance(other, Hamiltonian):
            # unfold parenthesis
            for key1 in self.elements.keys():
                for key2 in other.elements.keys():
                    phase, new_key = multiply_sets(key2, key1)
                    if new_key in out.elements.keys():
                        out[new_key] += phase * self[key1] * other[key2]
                    else: 
                        out[new_key] = phase * self[key1] * other[key2]
        elif isinstance(other, PauliOperator):
            key2 = ((other.qubit_id, other.name), )
            for key1 in self.elements.keys():
                phase, new_key = multiply_sets(key2, key1)
                if new_key in out.elements.keys():
                    out[new_key] += phase * self[key1]
                else: 
                    out[new_key] = phase * self[key1]
        elif isinstance(other, (int, float, complex)):
            out = copy.copy(self) 
            for key in out.elements.keys():
                out[key] *= other
            return out
        else:
            raise ValueError(f"invalid addition between Hamiltonian and {other.__class__.__name__}.")

        out.filter()
        return out

    def __rtruediv__(self, other):
        if not isinstance(other, (float, int, complex)):
            raise ValueError("Division of operators is not supported")
       
        return (1/other ) * self

    def __rfloordiv__(self, other):
        if not isinstance(other, (float, int, complex)):
            raise ValueError("Division of operators is not supported")
       
        return (1/other ) * self

    def __rsub__(self, other):
        return other + (-1 * self)
