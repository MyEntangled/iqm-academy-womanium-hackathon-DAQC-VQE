from qiskit.opflow.primitive_ops import PauliSumOp, PauliOp
from qiskit.quantum_info import Pauli
from qiskit.opflow import Z, Y, X, I

import numpy as np
import networkx as nx

from itertools import permutations


class SwapMap():
    '''
    Determine the qubit configuration at the beginning and swap layers from layer 2 onwards.
    '''
    def __init__(self, paulis):
        self.paulis = paulis
        self.pauli_strings = [pauli.primitive.__str__() for pauli in self.paulis]
        self.pauli_objs = [Pauli(string) for string in self.pauli_strings]
        self.coeffs = {self.pauli_strings[i]: abs(self.paulis[i].coeff) for i in range(len(self.paulis))}
        self.num_qubits = len(self.pauli_strings[0])

        self.possible_local_paulis = [X ^ X, X ^ Y, X ^ Z, Y ^ X, Y ^ Y, Y ^ Z, Z ^ X, Z ^ Y, Z ^ Z]
        self.possible_local_pauli_strings = [pauli.primitive.__str__() for pauli in self.possible_local_paulis]
        self.possible_local_pauli_objs = [Pauli(string) for string in self.possible_local_pauli_strings]

    def _connectivity_strength(self, qubit_1, qubit_2):
        self.local_coeffs = {}
        for i, string in enumerate(self.pauli_strings):
            H1 = string[qubit_1]
            H2 = string[qubit_2]
            local_H = ''.join([H1, H2])
            if local_H in self.possible_local_pauli_strings:

                if local_H in self.local_coeffs:
                    self.local_coeffs[local_H] += self.paulis[i].coeff
                else:
                    self.local_coeffs[local_H] = self.paulis[i].coeff

        local_paulis = [PauliOp(Pauli(term), coeff=coeff) for term, coeff in self.local_coeffs.items()]

        if len(local_paulis) == 0:
            return 0
        return np.linalg.norm(sum(local_paulis).to_matrix(), ord=2)

    def _Hamiltonian_graph(self):
        edges = []
        for qubit_1 in range(self.num_qubits):
            for qubit_2 in range(qubit_1+1, self.num_qubits):
                strength = self._connectivity_strength(qubit_1, qubit_2)
                if strength > 0:
                    edges.append((qubit_1, qubit_2, strength))

        G = nx.Graph()
        G.add_nodes_from(np.arange(0, self.num_qubits))
        G.add_weighted_edges_from(edges)
        # plot_graph(G)
        return G

    def _create_path_connectivity_graph(self, is_cycle):
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, self.num_qubits))
        for i in range(self.num_qubits):
            if (i == self.num_qubits - 1) & (is_cycle == True):
                G.add_edge(0, i)
            elif (i == self.num_qubits - 1) & (is_cycle == False):
                continue
            else:
                G.add_edge(i, i + 1)
        # plot_graph(G)
        return G

    def _create_grid_connectivity_graph(self, grid_size):
        width, length = grid_size
        device_num_qubits = width * length
        assert device_num_qubits == self.num_qubits

        grid = np.arange(0, device_num_qubits).reshape(width, length)

        G = nx.Graph()
        G.add_nodes_from(np.arange(0,device_num_qubits))
        for i in range(width):
            G.add_edges_from([(grid[i,j],grid[i,j+1]) for j in range(length-1)])

        for j in range(length):
            G.add_edges_from([(grid[i,j],grid[i+1,j]) for i in range(width-1)])

        return G


    def _find_optimal_map_brute_force(self, connectivity_graph):
        H_graph = self._Hamiltonian_graph()
        edges = list(connectivity_graph.edges())
        edge_perms = list(permutations(range(self.num_qubits)))

        s_max = 0
        mapping = ()
        for edge_perm in edge_perms:
            s_tmp = 0
            for edge in edges:
                if H_graph.has_edge(edge_perm[edge[0]], edge_perm[edge[1]]):
                    s_tmp += H_graph.get_edge_data(edge_perm[edge[0]], edge_perm[edge[1]])['weight']

            if s_tmp > s_max:
                s_max = s_tmp
                mapping = edge_perm

        return mapping, s_max

    def _count_swaps(self, mapping, connectivity_graph=None):
        if connectivity_graph is None:
            connectivity_graph = self._create_path_connectivity_graph(is_cycle=True)

        node_list = list(connectivity_graph.nodes())

        new_map = list(mapping).copy()
        swap_map = {}


        for j in range(len(node_list) - 1):
            cost = 0
            if new_map[j] != list(connectivity_graph.nodes())[j]:
                for k in range(j + 1, len(node_list)):
                    if new_map[k] == list(connectivity_graph.nodes())[j]:
                        cost = nx.shortest_path_length(connectivity_graph, new_map[j], new_map[k])
                        if cost != 1:
                            cost = cost * 2 - 1
                        swap_map[(new_map[j], new_map[k])] = cost
                        new_map[j], new_map[k] = list(new_map)[k], list(new_map)[j]
        return swap_map

    def reorder_qubits(self, map_type, grid_size=None,):
        connectivity_graph = None
        if map_type == 'linear':
            connectivity_graph = self._create_path_connectivity_graph(is_cycle=False)
        elif map_type == 'circular':
            connectivity_graph = self._create_path_connectivity_graph(is_cycle=True)
        elif map_type == 'grid':
            connectivity_graph = self._create_grid_connectivity_graph(grid_size)

        qubits_order, map_weight = self._find_optimal_map_brute_force(connectivity_graph)
        swap_map = self._count_swaps(qubits_order, connectivity_graph)

        return qubits_order, swap_map