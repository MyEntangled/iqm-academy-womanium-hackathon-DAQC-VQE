import numpy as np
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.opflow.primitive_ops import PauliOp

from qiskit.circuit import Parameter

import warnings

class ZX_Cycle():
    '''
    Create evolutions regarding to the ZX periodical connectivity
    with Ha = - sum(Z) + sum(Z{i}^X_{i+1})
    '''
    def __init__(self, num_qubits, connection_strength=None):
        self.num_qubits = num_qubits

        self.H_map_layout = []
        if self.num_qubits == 2:
            self.H_map_layout = ['ZX']
        else:
            for i in range(self.num_qubits-1):
                self.H_map_layout.append('I'*i + 'ZX' + 'I'*(self.num_qubits - i - 2))
            self.H_map_layout.append('X' + 'I'*(self.num_qubits - 2) + 'Z')

        ## Set up connectivity strength g_ij
        if connection_strength is None:
            self.connect_strength = [1]*len(self.H_map_layout)
        else:
            assert len(connection_strength) == len(self.H_map_layout)
            self.connect_strength = connection_strength

        self.H_map = []
        ## 1 body
        local_terms = ['I'*i + 'Z' + 'I'*(self.num_qubits-1-i) for i in range(self.num_qubits)]
        self.H_map += [PauliOp(Pauli(term), coeff=-1/np.sqrt(2)) for term in local_terms]
        ## 2 body
        self.H_map += [PauliOp(Pauli(pauli_string), coeff=self.connect_strength[i])
                      for i,pauli_string in enumerate(self.H_map_layout)]

    def _detect_interaction(self, pauli_string):
        '''
        Detect 2-body interactions (mean-field approximation may apply)
        '''
        non_identity_qubits = [qubit for qubit, pauli in enumerate(pauli_string) if pauli != 'I']
        if len(non_identity_qubits) < 2:
            return None
        elif len(non_identity_qubits) == 2:
            return non_identity_qubits
        else:
            warnings.warn("There is some >3-body terms!")


    def decoupling_rule(self, pauli_strings, clusters=None):
        '''
        Get decoupling rule, which is the operator D.
        '''
        if clusters:
            new_qubit_order = [qubit for cluster in clusters for qubit in cluster]
        else:
            new_qubit_order = range(self.num_qubits)

        connectivity_count = np.zeros((self.num_qubits, self.num_qubits))
        for pauli_string in pauli_strings:
            non_identity_qubits = self._detect_interaction(pauli_string)
            if non_identity_qubits is not None:
                connectivity_count[non_identity_qubits[0], non_identity_qubits[1]] = 1

        decoupling_pauli = [0] * self.num_qubits

        # Decoupling pauli according to new_qubit_order
        for i in range(len(new_qubit_order)):
            if connectivity_count[new_qubit_order[i], new_qubit_order[(i+1)%self.num_qubits]] == 0:
                decoupling_pauli[(i+1) % self.num_qubits] = 'Y'
            else:
                decoupling_pauli[(i+1) % self.num_qubits] = 'i'

        return ''.join(decoupling_pauli)

    def decoupled_time_evolution(self, pauli_strings, clusters=None):
        '''
        Create Ha' = D @ Ha @ D @ Ha, where D is the decoupling layer
        '''
        D = self.decoupling_rule(pauli_strings, clusters=clusters)
        Ha_prime_circ = QuantumCircuit(self.num_qubits, name='Ha')
        D_circ = QuantumCircuit(self.num_qubits)
        for qubit,pauli in enumerate(D):
            if pauli == 'X':
                D_circ.x(qubit)
            elif pauli == 'Y':
                D_circ.y(qubit)
            elif pauli == 'Z':
                D_circ.z(qubit)

        phi = Parameter('ϕ')
        Ha_prime_circ = Ha_prime_circ.compose(D_circ)
        Ha_prime_circ.hamiltonian(operator=sum(self.H_map), time=-phi, qubits=Ha_prime_circ.qubits, label=f'analog {clusters}')
        Ha_prime_circ = Ha_prime_circ.compose(D_circ)
        Ha_prime_circ.hamiltonian(operator=sum(self.H_map), time=-phi, qubits=Ha_prime_circ.qubits, label=f'analog {clusters}')

        return Ha_prime_circ