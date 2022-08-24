from Connectivity import ZX_Cycle
from SwapMap import SwapMap

from helper import convert_hamiltonian

import numpy as np

from qiskit.circuit.library import U3Gate
from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit

from qiskit.opflow import I

class Trotterizer():
    '''
    This has nothing to do with the conventional trotterization.
    It is rather a template to generate our ansatz.
    '''
    def __init__(self, num_qubits, paulis, connectivity_style):
        self.num_qubits = num_qubits
        self.paulis = paulis
        self.pauli_strings = [pauli.primitive.__str__() for pauli in self.paulis]

        self.U1 = lambda theta: U3Gate(theta-np.pi, np.pi, 0)
        self.U2 = lambda theta: U3Gate(np.pi, theta/2, -theta/2+np.pi)
        self.U3 = lambda theta: U3Gate(theta-np.pi, -np.pi/2, -np.pi/2)


        assert connectivity_style in ['ZX_cycle', 'ZX_grid']
        if connectivity_style == 'ZX_cycle':
            self.connectivity_map = ZX_Cycle(self.num_qubits, connection_strength=None)

        self.Ha_prime_circ = self.connectivity_map.decoupled_time_evolution(pauli_strings=self.pauli_strings, clusters=None)


    def _one_layer(self, layer_idx, Ha_prime_circ_layer=None):
        if Ha_prime_circ_layer is None:
            Ha_prime_circ_layer = self.Ha_prime_circ

        t = Parameter('t'+str(layer_idx))
        Ha_prime_bind = Ha_prime_circ_layer.assign_parameters([t])

        layer_circ = QuantumCircuit(self.num_qubits)
        a = ParameterVector('a'+str(layer_idx), self.num_qubits)
        b = ParameterVector('b'+str(layer_idx), self.num_qubits)
        c = ParameterVector('c'+str(layer_idx), self.num_qubits)
        d = ParameterVector('d'+str(layer_idx), self.num_qubits)

        ## RX RZ U2 U1
        for i in range(self.num_qubits):
            layer_circ.rx(theta=a[i], qubit=i)
        for i in range(self.num_qubits):
            layer_circ.rz(phi=b[i], qubit=i)

        for i in range(self.num_qubits):
            layer_circ.append(self.U2(c[i]), qargs=[i])
            layer_circ.append(self.U1(d[i]), qargs=[i])

        ## Ha'
        layer_circ = layer_circ.compose(Ha_prime_bind)

        ## U1 U2
        for i in range(self.num_qubits):
            layer_circ.append(self.U1(d[i]), qargs=[i])
            layer_circ.append(self.U2(c[i]), qargs=[i])

        return layer_circ

    def trotterize(self, num_layers):
        trotter_circuit = QuantumCircuit(self.num_qubits)

        for i in range(num_layers):
            layer_circ = self._one_layer(layer_idx=i)
            trotter_circuit = trotter_circuit.compose(layer_circ)
            #trotter_circuit.barrier()

        return trotter_circuit

    def _cluster_qubits(self, layer_order):
        ## layer_order also = stepsize - 1
        ## layer_order counts from 0

        clusters = []
        start = 0
        while start < layer_order+1:
            cluster = list(range(start, self.num_qubits, layer_order + 1))
            if len(cluster) > 0:
                clusters.append(cluster)
                start += 1
            else:
                break

        return clusters

    def _get_swaps_circs(self, swap_map):
        def adjacent_swap(q_start, step, rightward):
            circ = QuantumCircuit(self.num_qubits)

            if rightward == True:
                for i in range(step):
                    circ.swap((q_start+i)%self.num_qubits, (q_start+i+1)%self.num_qubits)
                for i in range(step-1, 0, -1):
                    circ.swap((q_start+i)%self.num_qubits, (q_start+i-1)%self.num_qubits)

            else:
                for i in range(step):
                    circ.swap((q_start-i)%self.num_qubits, (q_start-i-1)%self.num_qubits)
                for i in range(step-1, 0, -1):
                    circ.swap((q_start-i)%self.num_qubits, (q_start-i+1)%self.num_qubits)

            return circ
        ## The key in swap_map is based on the original qubit number, not position
        order = list(range(self.num_qubits))

        inv_swap_map = {qubits:num_swap for qubits,num_swap in reversed(list(swap_map.items()))} # map from naturally ordered list to some list
        inv_swaps_circ = QuantumCircuit(self.num_qubits)

        for (q1,q2), num_swaps in inv_swap_map.items():
            idx_q1 = order.index(q1)
            idx_q2 = order.index(q2)
            swap_circ = None

            if idx_q1 < idx_q2:
                len_to_right = idx_q2 - idx_q1
                len_to_left = idx_q1 + self.num_qubits - idx_q2

                if len_to_right <= len_to_left:# and 2*len_to_right-1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q1,step=len_to_right,rightward=True)
                elif len_to_left < len_to_right:# and 2*len_to_left-1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q1, step=len_to_left, rightward=False)
                else:
                    print('Something is wrong 1')

            elif idx_q1 > idx_q2:
                len_to_right = idx_q1 - idx_q2
                len_to_left = idx_q2 + self.num_qubits - idx_q1

                if len_to_right <= len_to_left:#and 2 * len_to_right - 1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q2, step=len_to_right, rightward=True)
                elif len_to_left < len_to_right:# and 2 * len_to_left - 1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q2, step=len_to_left, rightward=True)
                else:
                    print('Something is wrong 2')

            else:
                print('Something is wrong 3')
            order[idx_q1] = q2
            order[idx_q2] = q1
            inv_swaps_circ = inv_swaps_circ.compose(swap_circ)

        #rev_swaps = rev_swaps_circ.to_gate(label='multiswaps_rev')

        swaps_circ = inv_swaps_circ.inverse()

        return swaps_circ, inv_swaps_circ


    def multiorder_trotterize(self, num_layers):
        trotter_circuit = QuantumCircuit(self.num_qubits)

        for i in range(num_layers):
            if i == 0:
                layer_circ = self._one_layer(layer_idx=i)
                trotter_circuit = trotter_circuit.compose(layer_circ)

                trotter_circuit.barrier()
                trotter_circuit.barrier()

            else:
                clusters = self._cluster_qubits(layer_order=i)
                Ha_prime_circ = self.connectivity_map.decoupled_time_evolution(pauli_strings=self.pauli_strings,
                                                                                clusters=clusters)
                layer_circ = self._one_layer(layer_idx=i, Ha_prime_circ_layer=Ha_prime_circ)

                qubit_order = [qubit for cluster in clusters for qubit in cluster]
                swap_map = SwapMap([I^self.num_qubits])._count_swaps(mapping=qubit_order)

                # print(qubit_order)
                # print(swap_map)
                # print({k:v for k,v in reversed(list(swap_map.items()))})
                swaps_circ, inv_swaps_circ = self._get_swaps_circs(swap_map)

                trotter_circuit = trotter_circuit.compose(inv_swaps_circ)
                trotter_circuit.barrier()
                trotter_circuit = trotter_circuit.compose(layer_circ)
                trotter_circuit.barrier()
                trotter_circuit = trotter_circuit.compose(swaps_circ)

                trotter_circuit.barrier()
                trotter_circuit.barrier()

        return trotter_circuit

    def measure(self):
        pass