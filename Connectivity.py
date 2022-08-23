import numpy as np
from qiskit.quantum_info import Pauli
from qiskit import QuantumCircuit
from qiskit.opflow.primitive_ops import PauliOp

from qiskit.opflow import PauliTrotterEvolution, Suzuki
from qiskit.extensions import HamiltonianGate

from qiskit.circuit import Parameter

import warnings

import networkx as nx
from ast import literal_eval
from itertools import permutations
import copy

from qiskit.opflow import Z, Y, X, I



class ZX_Cycle():
    def __init__(self, num_qubits, connection_strength=None):
        self.num_qubits = num_qubits
        #self.coeffs = pauli_coeffs

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
        local_terms = ['I'*i + 'X' + 'I'*(self.num_qubits-1-i) for i in range(self.num_qubits)]
        self.H_map += [PauliOp(Pauli(term), coeff=-1./np.sqrt(2)) for term in local_terms]
        ## 2 body
        self.H_map += [PauliOp(Pauli(pauli_string), coeff=self.connect_strength[i])
                      for i,pauli_string in enumerate(self.H_map_layout)]
        # ## 3 body
        # threebody_terms = ['I'*i + 'YYY' + 'I'*(self.num_qubits-3-i) for i in range(self.num_qubits-2)]
        # threebody_terms.append('Y' + 'I'*(self.num_qubits-3) + 'YY')
        # threebody_terms.append('YY' + 'I'*(self.num_qubits-3) + 'Y')
        # self.H_map += [PauliOp(Pauli(term), coeff=0.05) for term in threebody_terms]


        phi = Parameter('ϕ')
        evolution_op = (-phi * sum(self.H_map)).exp_i()  # exp(iϕA)
        trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=1)).convert(evolution_op)

        ## Bind parameter when using Ha circuit, e.g. self.Ha_circ.assign_parameters([phi_value])
        self.Ha_circ = trotterized_op.to_circuit()
        #self.Ha_gate = HamiltonianGate(data=-phi * sum(self.H_map), time=1)

    def _detect_interaction(self, pauli_string):
        '''
        Assume pauli_string is 2-body
        :param pauli_string:
        :return:
        '''
        non_identity_qubits = [qubit for qubit, pauli in enumerate(pauli_string) if pauli != 'I']
        if len(non_identity_qubits) < 2:
            return None
        elif len(non_identity_qubits) == 2:
            return non_identity_qubits
        else:
            warnings.warn("There is some >3-body terms!")


    def decoupling_rule(self, pauli_strings):
        '''
        Assume H_map is a Hamiltonian on an underlying circular ZX-type connectivity.
        (Assume energy Hamiltonian is 2-body)
        :param H_map:
        :return:
        '''
        connectivity_count = np.zeros((self.num_qubits, self.num_qubits))
        for pauli_string in pauli_strings:
            non_identity_qubits = self._detect_interaction(pauli_string)
            if non_identity_qubits is not None:
                #connectivity_count[non_identity_qubits[0], non_identity_qubits[1]] += self.coeffs[pauli_string]
                connectivity_count[non_identity_qubits[0], non_identity_qubits[1]] = 1

        decoupling_pauli = [0] * self.num_qubits
        for i in range(self.num_qubits):
            if connectivity_count[i, (i + 1) % self.num_qubits] == 0:
                decoupling_pauli[(i + 1) % self.num_qubits] = 'Z'
            else:
                decoupling_pauli[(i + 1) % self.num_qubits] = 'I'

        return ''.join(decoupling_pauli)

    def decoupled_time_evolution(self, pauli_strings):
        D = self.decoupling_rule(pauli_strings)
        Ha_prime_circ = QuantumCircuit(self.num_qubits, name='Ha')
        D_circ = QuantumCircuit(self.num_qubits)
        for qubit,pauli in enumerate(D):
            if pauli == 'X':
                D_circ.x(qubit)
            elif pauli == 'Y':
                D_circ.y(qubit)
            elif pauli == 'Z':
                D_circ.z(qubit)

        Ha_prime_circ = Ha_prime_circ.compose(D_circ).compose(self.Ha_circ).compose(D_circ).compose(self.Ha_circ)
        # Ha_prime_circ = Ha_prime_circ.compose(D_circ)
        # Ha_prime_circ.unitary(self.Ha_gate, Ha_prime_circ.qubits, label='analog')
        # Ha_prime_circ = Ha_prime_circ.compose(D_circ)
        # Ha_prime_circ.unitary(self.Ha_gate, Ha_prime_circ.qubits, label='analog')

        return Ha_prime_circ


if __name__ == '__main__':
    H_e = [2 * X ^ I ^ I ^ I ^ I ^ X, 3.1 * 2 * I ^ Y ^ I ^ Y ^ I ^ I, 4.1 * I ^ I ^ I ^ Y ^ I ^ Z,
           5 * I ^ I ^ I ^ I ^ X ^ Z,
           3.4 * I ^ I ^ X ^ I ^ X ^ I, 3 * I ^ I ^ I ^ I ^ X ^ Z, 2.4 * I ^ X ^ I ^ I ^ I ^ Z,
           5 * I ^ X ^ I ^ I ^ X ^ I, 3.7 * X ^ I ^ X ^ I ^ I ^ I,
           2.4 * I ^ I ^ X ^ X ^ I ^ I, 3 * I ^ I ^ I ^ I ^ X ^ Z, 2.3 * I ^ X ^ I ^ I ^ X ^ I,
           5.2 * I ^ I ^ I ^ X ^ X ^ I]
    connectivity = ZX_Cycle(num_qubits=5)
    swap_map = connectivity.get_swap_map(H_e)
    print(swap_map)