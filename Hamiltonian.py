from qiskit.quantum_info import Pauli
from qiskit.opflow import Z, Y, X, I
from qiskit.opflow.primitive_ops import PauliSumOp, PauliOp

import numpy as np
import scipy.optimize

import pennylane as qml
import networkx as nx
from matplotlib import pyplot as plt

import warnings


class Hamiltonian():
    def __init__(self, paulis):
        ## Pauli terms present in the problem Hamiltonian
        self.paulis = paulis
        self.pauli_strings = [pauli.primitive.__str__() for pauli in self.paulis]
        self.pauli_objs = [Pauli(string) for string in self.pauli_strings]
        self.coeffs = {self.pauli_strings[i]: abs(self.paulis[i].coeff) for i in range(len(self.paulis))}
        self.cross_weights = np.outer(list(self.coeffs.values()), list(self.coeffs.values()))

        self.pauli_idx = {}
        for idx, string in enumerate(self.pauli_strings):
            self.pauli_idx[string] = idx

        self.commute_matrix, self.commute_list = self._commutation_info(self.paulis, self.pauli_strings, self.pauli_objs)
        self.cross_commute_score = self.cross_weights * self.commute_matrix
        self.cross_anticommute_score = self.cross_weights * (1-self.commute_matrix)

        ## All possible Two-local Pauli terms
        self.possible_local_paulis = [X ^ X, X ^ Y, X ^ Z, Y ^ X, Y ^ Y, Y ^ Z, Z ^ X, Z ^ Y, Z ^ Z]
        self.possible_local_pauli_strings = [pauli.primitive.__str__() for pauli in self.possible_local_paulis]
        self.possible_local_pauli_objs = [Pauli(string) for string in self.possible_local_pauli_strings]
        self.local_coeffs = None
        self.local_cross_wights = None

        self.local_pauli_idx = {}
        for idx, string in enumerate(self.possible_local_pauli_strings):
            self.local_pauli_idx[string] = idx

        self.local_commute_matrix, self.local_commute_list = self._commutation_info(self.possible_local_paulis,
                                                                                    self.possible_local_pauli_strings,
                                                                                    self.possible_local_pauli_objs)

        ## Prepare graph
        self.G = nx.Graph()
        self.G.add_nodes_from(self.pauli_strings)
        self.G.add_edges_from(self.commute_list)
        self.C = nx.complement(self.G)

        ## Miscel
        self.num_qubits = len(self.pauli_strings[0])


    def _commutation_info(self, paulis, pauli_strings, pauli_objs):
        '''
        Prepare commute matrix and commute list (When a class instance is created)
        '''
        commute_matrix = np.eye(len(paulis))  # 1 for commute, #0 for anticommute
        for i in range(len(paulis)):
            for j in range(i + 1, len(paulis)):
                if pauli_objs[i].commutes(pauli_objs[j]):
                    commute_matrix[i, j] = 1
                    commute_matrix[j, i] = 1

        commute_list = []
        for i in range(len(paulis)):
            for j in range(i + 1, len(paulis)):
                if commute_matrix[i, j] == 1:
                    commute_list.append([pauli_strings[i], pauli_strings[j]])

        return commute_matrix, commute_list

    def _find_clique_covers(self, C_graph):
        '''
        Find clique covering for G (C_graph is G.complement)
        Return a few number of clique covers such that each cover has the smallest (and the same) number of cliques.
        '''
        all_strategies = ['largest_first', 'random_sequential', 'smallest_last', 'independent_set',
                          'connected_sequential_bfs', 'connected_sequential_dfs', 'DSATUR']
        strategy_groups = [nx.coloring.greedy_color(C_graph, strategy=each) for each in all_strategies]

        ## Count size of groupings and select a grouping with smallest number of groups
        strategy_num_groups = [len(set(groups.values())) for groups in strategy_groups]
        print("Minimum number of commuting groupings found:", strategy_num_groups)
        min_num_groups = min(strategy_num_groups)
        strategy_groups = [groups for groups in strategy_groups if len(set(groups.values())) == min_num_groups]

        ## Remove duplicated groupings
        temp = []
        for groups in strategy_groups:
            check_dup = False
            for non_dup_groups in temp:
                if groups == non_dup_groups:
                    check_dup = True
            if not check_dup:
                temp.append(groups)
        strategy_groups = temp

        ## Convert to list of families
        strategy_groups_idx = [set(groups.values()) for groups in strategy_groups]
        strategy_families = [
            [[term for term, group_idx in strategy_groups[i].items() if group_idx == val] for val in strategy_groups_idx[i]] for i
            in range(len(strategy_groups))]

        return strategy_families

    def _cross_family_relation_score(self, families, commute_relation:bool):
        if commute_relation:
            cross_relation_matrix = self.commute_matrix
        else:
            cross_relation_matrix = 1-self.commute_matrix

        family_relation_score = np.zeros((len(families), len(families)))


        for i in range(len(families)):
            paulis_i_idx = [self.pauli_idx[pauli] for pauli in families[i]]

            for j in range(len(families)):
                paulis_j_idx = [self.pauli_idx[pauli] for pauli in families[j]]
                family_relation_score[i, j] = np.sum(self.cross_weights[paulis_i_idx, :][:, paulis_j_idx]
                                                        * cross_relation_matrix[paulis_i_idx, :][:, paulis_j_idx])

        family_relation_score[range(len(families)), range(len(families))] = 0

        return family_relation_score

    def _clique_ordering(self, cliques, maximize_commute:bool):
        '''
        Return an ordering of cliques (or families) to maximize the total commute score between consecutive cliques.
        (Equivalently, minimze total anticommute score --> Traveling Salesman Problem)

        Find Hamiltonian cycle with maximum edge weights with an approximation using minimum spanning tree
        https://student.cs.uwaterloo.ca/~cs466/Old_courses/F15/5-Approximation.pdf
        '''
        clique_commute_score = self._cross_family_relation_score(cliques, commute_relation=True)
        clique_anticommute_score = self._cross_family_relation_score(cliques, commute_relation=False)

        clique_graph = nx.Graph()
        clique_graph.add_nodes_from(range(len(cliques)))
        if maximize_commute == True:
            for i in range(len(cliques)):
                for j in range(len(cliques)):
                    if j > i:
                        clique_graph.add_edge(i, j, weight=clique_anticommute_score[i, j])
        else:
            for i in range(len(cliques)):
                for j in range(len(cliques)):
                    if j > i:
                        clique_graph.add_edge(i, j, weight=clique_commute_score[i, j])

        ordering = nx.algorithms.approximation.traveling_salesman_problem(clique_graph, cycle=False)
        ordered_cliques = [cliques[index] for index in ordering]

        total_score = 0
        if maximize_commute == True:
            for i in range(len(ordering)-1):
                total_score += clique_commute_score[ordering[i], ordering[i+1]]
        else:
            for i in range(len(ordering)-1):
                total_score += clique_anticommute_score[ordering[i], ordering[i+1]]


        #         TODO

        #         ## Assign first and last term of each clique
        #         interclique_commute = []
        # #        commute_to_prev_clique = []
        #         for i in range(len(ordered_cliques)):
        #             curr_clique = ordered_cliques[i]
        #             next_clique = ordered_cliques[(i+1)%len(ordered_cliques)]
        #             temp = []
        #             for curr_pauli in curr_clique:
        #                 curr_pauli_idx = self.pauli_idx[curr_pauli]
        #                 for next_pauli in next_clique:
        #                     next_pauli_idx = self.pauli_idx[next_pauli]
        #                     if self.commute_matrix[curr_pauli_idx, next_pauli_idx] == 1:

        #                         temp.append((curr_pauli, next_pauli, cross_weights[curr_pauli_idx,next_pauli_idx]))
        #             interclique_commute.append(temp)

        return ordered_cliques, total_score


    def find_maximal_commuting_cliques(self):
        '''
        Find a minimal clique covering, i.e. cliques are expected to be as large as possible
        :return:
        '''
        strategy_families = self._find_clique_covers(C_graph=self.C)

        strategy_optimal_ordering = []
        strategy_total_score = []

        for families in strategy_families:
            ordered_families, total_commute_score = self._clique_ordering(families, maximize_commute=True)
            strategy_optimal_ordering.append(ordered_families)
            strategy_total_score.append(total_commute_score)


        index = np.argmin(strategy_total_score)
        ordered_cliques = strategy_families[index]
        total_score = strategy_total_score[index]
        return ordered_cliques, total_score


    def select_H0(self):
        '''
        Select H0 as a set that is intra-commuting and maximally inter-anticommuting
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.102.4678&rep=rep1&type=pdf
        '''

        ## A_ub @ x <= b_ub
        A_ub = np.zeros((self.C.number_of_edges(), len(self.paulis)))
        b_ub = np.ones(self.C.number_of_edges())
        for i, edge in enumerate(self.C.edges()):
            A_ub[i, self.pauli_idx[edge[0]]] = 1
            A_ub[i, self.pauli_idx[edge[1]]] = 1
        # print(A_ub)

        # Anticommute score from each pauli string towards everyone else
        # score = np.zeros(len(self.pauli_strings))
        # for i, string_i in enumerate(self.pauli_strings):
        #     for j, string_j in enumerate(self.pauli_strings):
        #         if j > i and self.commute_matrix[i,j] == 0:
        #             score[i] += self.coeffs[string_i] * self.coeffs[string_j]
        #             score[j] += self.coeffs[string_j] * self.coeffs[string_i]

        res = scipy.optimize.linprog(c=-np.sum(self.cross_anticommute_score, axis=0), A_ub=A_ub, b_ub=b_ub, bounds=(0, 1))
        x = res.x

        S_zero = []
        S_half = []
        S_one = []
        for idx, val in enumerate(x):
            if abs(val - 0) < 0.05:
                S_zero.append(self.pauli_strings[idx])
            elif abs(val - 0.5) < 0.05:
                S_half.append(self.pauli_strings[idx])
            else:
                S_one.append(self.pauli_strings[idx])

        ## Get an independent graph in the induced graph given by S_half
        subgraph = nx.induced_subgraph(self.C, S_half).copy()
        selected_from_S_half = []

        while len(subgraph) > 0:
            max_pauli = min(subgraph.nodes(), key=lambda x: self.coeffs[x])
            selected_from_S_half.append(max_pauli)

            to_be_removed = list(subgraph.adj[max_pauli]) + [max_pauli]
            subgraph.remove_nodes_from(to_be_removed)

        ### This set is intra-commuting and maximally inter-anticommuting
        max_anticommute_set = S_one + selected_from_S_half
        return max_anticommute_set


    # def connectivity_strength(self, qubit_1, qubit_2):
    #     self.local_coeffs = dict(zip(self.local_paulis, [0] * len(self.local_paulis)))
    #     for string in H.pauli_strings:
    #         H1 = string[qubit_1]
    #         H2 = string[qubit_2]
    #         local_H = ''.join([H1, H2])
    #         if local_H in self.local_paulis:
    #             self.local_coeffs[local_H] += self.coeffs[string]
    #
    #     # print(local_coeffs)
    #
    #     local_G = nx.Graph()
    #     local_G.add_nodes_from([local for local, coeff in self.local_coeffs.items() if coeff > 0])
    #
    #     local_edges = [edge for edge in self.local_commute_list
    #                    if (self.local_coeffs[edge[0]] > 0 and self.local_coeffs[edge[1]] > 0)]
    #     print(local_edges)
    #     local_G.add_edges_from(local_edges)
    #
    #     local_C = nx.complement(local_G)
    #
    #
    #     strategy_families = self._find_clique_covers(C_graph=local_C)
    #
    #     strategy_optimal_ordering = []
    #     strategy_total_score = []
    #
    #     for families in strategy_families:
    #         ordered_families, total_anticommute_score = self._clique_ordering(families, maximize_commute=False)
    #         strategy_optimal_ordering.append(ordered_families)
    #         strategy_total_score.append(total_anticommute_score)
    #
    #     index = np.argmin(strategy_total_score)
    #     ordered_cliques = strategy_families[index]
    #     total_score = strategy_total_score[index]
    #
    #
    #     for families in strategy_families:
    #         clique_anticommute_score = self._cross_family_relation_score(families, commute_relation=False)
    #
    #
    #     return best_families

    def connectivity_strength(self, qubit_1, qubit_2):

        self.local_coeffs = {}
        for i,string in enumerate(self.pauli_strings):
            H1 = string[qubit_1]
            H2 = string[qubit_2]
            local_H = ''.join([H1, H2])
            if local_H in self.possible_local_pauli_strings:

                if local_H in self.local_coeffs:
                    self.local_coeffs[local_H] += self.paulis[i].coeff
                else:
                    self.local_coeffs[local_H] = self.paulis[i].coeff

        local_paulis = [PauliOp(Pauli(term), coeff=coeff) for term,coeff in self.local_coeffs.items()]

        return np.linalg.norm(sum(local_paulis).to_matrix(), ord=2)


if __name__ == '__main__':
    paulis = [2 * X ^ X ^ I ^ I, 3.1 * Y ^ Y ^ I ^ I, 4.1 * Z ^ Z ^ I ^ I, 5 * X ^ Y ^ I ^ I,
              3.4 * X ^ Z ^ I ^ I, 3 * Y ^ X ^ I ^ I, 2.4 * Y ^ Z ^ I ^ I, 5 * Z ^ X ^ I ^ I,
              3.7 * Z ^ Y ^ I ^ I]
    paulis = [2 * X ^ X ^ I ^ I, 3.1 * I ^ Y ^ Z ^ I, 4.1 * I ^ I ^ I ^ X, 5 * I ^ I ^ Y ^ Z,
              3.4 * X ^ I ^ I ^ I, 3 * I ^ I ^ X ^ Y, 2.4 * I ^ Z ^ Z ^ I, 5 * I ^ I ^ X ^ I,
              3.7 * Z ^ I ^ I ^ I]

    paulis = [1*X^X, 2*Y^Y, 3*Z^Z, 4*X^Z]
    H = Hamiltonian(paulis)
    best_families, total_score = H.find_maximal_commuting_cliques()
    H0 = H.select_H0()

    print(best_families, total_score)
    print(H0)

    # H_map = ['ZXII', 'IZXI', 'IIZX', 'XIIZ']
    # cliques_decoupling_pauli = [H.decoupling_rule(H_map, family) for family in best_families]
    # print('cliques', best_families)
    # print(cliques_decoupling_pauli)

    print(H.connectivity_strength(qubit_1=0, qubit_2=1))


