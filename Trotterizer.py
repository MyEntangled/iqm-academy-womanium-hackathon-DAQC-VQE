from Hamiltonian import Hamiltonian
from Connectivity import ZX_Cycle
from helper import convert_hamiltonian

import numpy as np

from qiskit.circuit.library import U3Gate
from qiskit.circuit import Parameter, ParameterVector
from qiskit import QuantumCircuit

class Trotterizer():
    def __init__(self, num_qubits, paulis, connectivity_style):
        self.num_qubits = num_qubits
        #self.H = Hamiltonian(paulis=paulis)
        self.paulis = paulis
        self.pauli_strings = [pauli.primitive.__str__() for pauli in self.paulis]

        self.U1 = lambda theta: U3Gate(theta-np.pi, np.pi, 0)
        self.U2 = lambda theta: U3Gate(np.pi, theta/2, -theta/2+np.pi)
        self.U3 = lambda theta: U3Gate(theta-np.pi, -np.pi/2, -np.pi/2)

        #self.cliques, _ = self.H.find_maximal_commuting_cliques()
        #self.H0 = self.H.select_H0()


        assert connectivity_style in ['ZX_cycle', 'ZX_grid']
        if connectivity_style == 'ZX_cycle':
            self.connectivity_map = ZX_Cycle(self.num_qubits, connection_strength=None)

        self.Ha_prime_circ = self.connectivity_map.decoupled_time_evolution(pauli_strings=self.pauli_strings, clusters=None)


    def _one_layer(self, layer_idx):
        t = Parameter('t'+str(layer_idx))
        Ha_prime_bind = self.Ha_prime_circ.assign_parameters([t])

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
            else:
                break

        return clusters

    def _add_swap(self, circ, swap_map):
        def adjacent_swap(q_start, step, rightward):
            swap_circ = QuantumCircuit(self.num_qubits)

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
        rev_swap_map = {qubits:num_swap for qubits,num_swap in reversed(list(swap_map.items()))} # map from naturally ordered list to some list
        rev_swaps_circ = QuantumCircuit(self.num_qubits)

        for (q1,q2), num_swaps in swap_map.items():
            idx_q1 = order.index(q1)
            idx_q2 = order.index(q2)
            order[idx_q1] = q2
            order[idx_q2] = q1

            if idx_q1 < idx_q2:
                len_to_right = idx_q2 - idx_q1
                len_to_left = idx_q1 + self.num_qubits - idx_q2

                if len_to_right <= len_to_left and 2*len_to_right-1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q1,step=len_to_right,rightward=True)
                elif len_to_left < len_to_right and 2*len_to_left-1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q1, step=len_to_right, rightward=False)


            elif idx_q1 > idx_q2:
                len_to_right = idx_q1 - idx_q2
                len_to_left = idx_q2 + self.num_qubits - idx_q1

                if len_to_right <= len_to_left and 2 * len_to_right - 1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q2, step=len_to_right, rightward=True)
                elif len_to_left < len_to_right and 2 * len_to_left - 1 == num_swaps:
                    swap_circ = adjacent_swap(q_start=idx_q2, step=len_to_right, rightward=False)

            else:
                pass
            rev_swaps_circ = 1


    def multiorder_trotterize(self, num_layers, swap_map=None):
        trotter_circuit = QuantumCircuit(self.num_qubits)

        if swap_map is not None:
            for (q1,q2),num_swaps in swap_map.items():
                trotter_circuit.swap(q1,q2)

        for i in range(num_layers):
            if i == 0:
                layer_circ = self._one_layer(layer_idx=i)
                trotter_circuit = trotter_circuit.compose(layer_circ)
            else:
                clusters = self._cluster_qubits(layer_order=i)
                layer_circ = self._one_layer(layer_idx=i)
                trotter_circuit = trotter_circuit.compose(layer_circ)

        return trotter_circuit

    def measure(self):
        pass


if __name__ == '__main__':
    from qiskit.opflow import Z, Y, X, I
    num_qubits = 10
    # paulis = [2 * X ^ X ^ I ^ I, 3.1 * I ^ Y ^ Z ^ I, 4.1 * I ^ I ^ I ^ X, 5 * I ^ I ^ Y ^ Z,
    #           3.4 * X ^ I ^ I ^ I, 3 * I ^ I ^ X ^ Y, 2.4 * I ^ Z ^ Z ^ I, 5 * I ^ I ^ X ^ I,
    #           3.7 * Z ^ I ^ I ^ I]

    # paulis = [-1.052373245772859 * I ^ I,
    #           0.39793742484318045 * I ^ Z,
    #           -0.39793742484318045 * Z ^ I,
    #           -0.01128010425623538 * Z ^ Z,
    #           0.18093119978423156 * X ^ X]

    input_hamiltonian = """
    + 0.003034656830204855 * IIIYYIIIYY

    + 0.003034656830204855 * IIIXXIIIYY

    + 0.003034656830204855 * IIIYYIIIXX

    + 0.003034656830204855 * IIIXXIIIXX

    - 0.008373361424264817 * YZZZYIIIYY

    - 0.008373361424264817 * XZZZXIIIYY

    - 0.008373361424264817 * YZZZYIIIXX

    - 0.008373361424264817 * XZZZXIIIXX

    + 0.00211113766859809 * YZZYIIIIYY

    + 0.00211113766859809 * XZZXIIIIYY

    + 0.00211113766859809 * YZZYIIIIXX

    + 0.00211113766859809 * XZZXIIIIXX

    - 0.00491756976241806 * IIIIIIIIYY

    - 0.00491756976241806 * IIIIIIIIXX

    + 0.010540187409026488 * ZIIIIIIIYY

    + 0.010540187409026488 * ZIIIIIIIXX

    - 0.0011822832324725804 * IZIIIIIIYY

    - 0.0011822832324725804 * IZIIIIIIXX

    - 0.0011822832324725804 * IIZIIIIIYY

    - 0.0011822832324725804 * IIZIIIIIXX

    - 0.00154067008970742 * IIIZIIIIYY

    - 0.00154067008970742 * IIIZIIIIXX

    + 0.011733623912074194 * IIIIZIIIYY

    + 0.011733623912074194 * IIIIZIIIXX

    + 0.0027757462269049522 * IIIIIZIIYY

    + 0.0027757462269049522 * IIIIIZIIXX

    + 0.0036202487558837123 * IIIIIIZIYY

    + 0.0036202487558837123 * IIIIIIZIXX

    + 0.0036202487558837123 * IIIIIIIZYY

    + 0.0036202487558837123 * IIIIIIIZXX

    + 0.005996760849734561 * IIYZYIIYZY

    + 0.005996760849734561 * IIXZXIIYZY

    + 0.005996760849734561 * IIYZYIIXZX

    + 0.005996760849734561 * IIXZXIIXZX

    + 0.004802531988356293 * IIYYIIIYZY

    + 0.004802531988356293 * IIXXIIIYZY

    + 0.004802531988356293 * IIYYIIIXZX

    + 0.004802531988356293 * IIXXIIIXZX

    - 0.004879740484191497 * YZYIIIIYZY

    - 0.004879740484191497 * XZXIIIIYZY

    - 0.004879740484191497 * YZYIIIIXZX

    - 0.004879740484191497 * XZXIIIIXZX

    + 0.005996760849734561 * IYZZYIYZZY

    + 0.005996760849734561 * IXZZXIYZZY

    + 0.005996760849734561 * IYZZYIXZZX

    + 0.005996760849734561 * IXZZXIXZZX

    + 0.004802531988356293 * IYZYIIYZZY

    + 0.004802531988356293 * IXZXIIYZZY

    + 0.004802531988356293 * IYZYIIXZZX

    + 0.004802531988356293 * IXZXIIXZZX

    - 0.004879740484191497 * YYIIIIYZZY

    - 0.004879740484191497 * XXIIIIYZZY

    - 0.004879740484191497 * YYIIIIXZZX

    - 0.004879740484191497 * XXIIIIXZZX

    - 0.008373361424264817 * IIIYYYZZZY

    - 0.008373361424264817 * IIIXXYZZZY

    - 0.008373361424264817 * IIIYYXZZZX

    - 0.008373361424264817 * IIIXXXZZZX

    + 0.0307383271773138 * YZZZYYZZZY

    + 0.0307383271773138 * XZZZXYZZZY

    + 0.0307383271773138 * YZZZYXZZZX

    + 0.0307383271773138 * XZZZXXZZZX

    - 0.0077644411821215335 * YZZYIYZZZY

    - 0.0077644411821215335 * XZZXIYZZZY

    - 0.0077644411821215335 * YZZYIXZZZX

    - 0.0077644411821215335 * XZZXIXZZZX

    - 0.005949019975734247 * IIIIIYZZZY

    - 0.005949019975734247 * IIIIIXZZZX

    - 0.0351167704024114 * ZIIIIYZZZY

    - 0.0351167704024114 * ZIIIIXZZZX

    + 0.0027298828353264134 * IZIIIYZZZY

    + 0.0027298828353264134 * IZIIIXZZZX

    + 0.0027298828353264134 * IIZIIYZZZY

    + 0.0027298828353264134 * IIZIIXZZZX

    + 0.0023679368995844726 * IIIZIYZZZY

    + 0.0023679368995844726 * IIIZIXZZZX

    - 0.03305872858775587 * IIIIZYZZZY

    - 0.03305872858775587 * IIIIZXZZZX

    - 0.0021498576488650843 * IIIIIYIZZY

    - 0.0021498576488650843 * IIIIIXIZZX

    - 0.0021498576488650843 * IIIIIYZIZY

    - 0.0021498576488650843 * IIIIIXZIZX

    + 0.004479074568182561 * IIIIIYZZIY

    + 0.004479074568182561 * IIIIIXZZIX

    + 0.004802531988356293 * IIYZYIIYYI

    + 0.004802531988356293 * IIXZXIIYYI

    + 0.004802531988356293 * IIYZYIIXXI

    + 0.004802531988356293 * IIXZXIIXXI

    + 0.010328819322301622 * IIYYIIIYYI

    + 0.010328819322301622 * IIXXIIIYYI

    + 0.010328819322301622 * IIYYIIIXXI

    + 0.010328819322301622 * IIXXIIIXXI

    - 0.003466391848475337 * YZYIIIIYYI

    - 0.003466391848475337 * XZXIIIIYYI

    - 0.003466391848475337 * YZYIIIIXXI

    - 0.003466391848475337 * XZXIIIIXXI

    + 0.004802531988356293 * IYZZYIYZYI

    + 0.004802531988356293 * IXZZXIYZYI

    + 0.004802531988356293 * IYZZYIXZXI

    + 0.004802531988356293 * IXZZXIXZXI

    + 0.010328819322301622 * IYZYIIYZYI

    + 0.010328819322301622 * IXZXIIYZYI

    + 0.010328819322301622 * IYZYIIXZXI

    + 0.010328819322301622 * IXZXIIXZXI

    - 0.003466391848475337 * YYIIIIYZYI

    - 0.003466391848475337 * XXIIIIYZYI

    - 0.003466391848475337 * YYIIIIXZXI

    - 0.003466391848475337 * XXIIIIXZXI

    + 0.00211113766859809 * IIIYYYZZYI

    + 0.00211113766859809 * IIIXXYZZYI

    + 0.00211113766859809 * IIIYYXZZXI

    + 0.00211113766859809 * IIIXXXZZXI

    - 0.0077644411821215335 * YZZZYYZZYI

    - 0.0077644411821215335 * XZZZXYZZYI

    - 0.0077644411821215335 * YZZZYXZZXI

    - 0.0077644411821215335 * XZZZXXZZXI

    + 0.006575744899182541 * YZZYIYZZYI

    + 0.006575744899182541 * XZZXIYZZYI

    + 0.006575744899182541 * YZZYIXZZXI

    + 0.006575744899182541 * XZZXIXZZXI

    + 0.023557442395837284 * IIIIIYZZYI

    + 0.023557442395837284 * IIIIIXZZXI

    + 0.010889407716094479 * ZIIIIYZZYI

    + 0.010889407716094479 * ZIIIIXZZXI

    - 0.0003518893528389501 * IZIIIYZZYI

    - 0.0003518893528389501 * IZIIIXZZXI

    - 0.0003518893528389501 * IIZIIYZZYI

    - 0.0003518893528389501 * IIZIIXZZXI

    - 0.00901204279263803 * IIIZIYZZYI

    - 0.00901204279263803 * IIIZIXZZXI

    + 0.0127339139792953 * IIIIZYZZYI

    + 0.0127339139792953 * IIIIZXZZXI

    - 0.003818281201314288 * IIIIIYIZYI

    - 0.003818281201314288 * IIIIIXIZXI

    - 0.003818281201314288 * IIIIIYZIYI

    - 0.003818281201314288 * IIIIIXZIXI

    + 0.004217284878422759 * IYYIIIYYII

    + 0.004217284878422759 * IXXIIIYYII

    + 0.004217284878422759 * IYYIIIXXII

    + 0.004217284878422759 * IXXIIIXXII

    - 0.004879740484191498 * IIYZYYZYII

    - 0.004879740484191498 * IIXZXYZYII

    - 0.004879740484191498 * IIYZYXZXII

    - 0.004879740484191498 * IIXZXXZXII

    - 0.003466391848475337 * IIYYIYZYII

    - 0.003466391848475337 * IIXXIYZYII

    - 0.003466391848475337 * IIYYIXZXII

    - 0.003466391848475337 * IIXXIXZXII

    + 0.004868302545087521 * YZYIIYZYII

    + 0.004868302545087521 * XZXIIYZYII

    + 0.004868302545087521 * YZYIIXZXII

    + 0.004868302545087521 * XZXIIXZXII

    - 0.004879740484191498 * IYZZYYYIII

    - 0.004879740484191498 * IXZZXYYIII

    - 0.004879740484191498 * IYZZYXXIII

    - 0.004879740484191498 * IXZZXXXIII

    - 0.003466391848475337 * IYZYIYYIII

    - 0.003466391848475337 * IXZXIYYIII

    - 0.003466391848475337 * IYZYIXXIII

    - 0.003466391848475337 * IXZXIXXIII

    + 0.004868302545087521 * YYIIIYYIII

    + 0.004868302545087521 * XXIIIYYIII

    + 0.004868302545087521 * YYIIIXXIII

    + 0.004868302545087521 * XXIIIXXIII

    - 0.004917569762418068 * IIIYYIIIII

    - 0.004917569762418068 * IIIXXIIIII

    + 0.0027757462269049522 * ZIIYYIIIII

    + 0.0027757462269049522 * ZIIXXIIIII

    + 0.0036202487558837123 * IZIYYIIIII

    + 0.0036202487558837123 * IZIXXIIIII

    + 0.0036202487558837123 * IIZYYIIIII

    + 0.0036202487558837123 * IIZXXIIIII

    - 0.005949019975734285 * YZZZYIIIII

    - 0.005949019975734285 * XZZZXIIIII

    - 0.0021498576488650843 * YIZZYIIIII

    - 0.0021498576488650843 * XIZZXIIIII

    - 0.0021498576488650843 * YZIZYIIIII

    - 0.0021498576488650843 * XZIZXIIIII

    + 0.004479074568182561 * YZZIYIIIII

    + 0.004479074568182561 * XZZIXIIIII

    + 0.02355744239583729 * YZZYIIIIII

    + 0.02355744239583729 * XZZXIIIIII

    - 0.003818281201314288 * YIZYIIIIII

    - 0.003818281201314288 * XIZXIIIIII

    - 0.003818281201314288 * YZIYIIIIII

    - 0.003818281201314288 * XZIXIIIIII

    + 1.0709274663656798 * IIIIIIIIII

    - 0.5772920990654371 * ZIIIIIIIII

    - 0.4244817531727133 * IZIIIIIIII

    + 0.06245512523136934 * ZZIIIIIIII

    - 0.4244817531727134 * IIZIIIIIII

    + 0.06245512523136934 * ZIZIIIIIII

    + 0.06558452315458405 * IZZIIIIIII

    - 0.3899177647415215 * IIIZIIIIII

    + 0.053929860773588405 * ZIIZIIIIII

    + 0.06022550139954594 * IZIZIIIIII

    + 0.06022550139954594 * IIZZIIIIII

    + 0.004360552555030484 * YZZYZIIIII

    + 0.004360552555030484 * XZZXZIIIII

    - 0.30101532158947975 * IIIIZIIIII

    + 0.08360121967246183 * ZIIIZIIIII

    + 0.06278876343471208 * IZIIZIIIII

    + 0.06278876343471208 * IIZIZIIIII

    + 0.053621410722614865 * IIIZZIIIII

    + 0.010540187409026488 * IIIYYZIIII

    + 0.010540187409026488 * IIIXXZIIII

    - 0.0351167704024114 * YZZZYZIIII

    - 0.0351167704024114 * XZZZXZIIII

    + 0.010889407716094479 * YZZYIZIIII

    + 0.010889407716094479 * XZZXIZIIII

    - 0.5772920990654372 * IIIIIZIIII

    + 0.11409163501020725 * ZIIIIZIIII

    + 0.06732342777645686 * IZIIIZIIII

    + 0.06732342777645686 * IIZIIZIIII

    + 0.060505605672770954 * IIIZIZIIII

    + 0.11433954684977561 * IIIIZZIIII

    - 0.0011822832324725804 * IIIYYIZIII

    - 0.0011822832324725804 * IIIXXIZIII

    + 0.0027298828353264134 * YZZZYIZIII

    + 0.0027298828353264134 * XZZZXIZIII

    - 0.0003518893528389501 * YZZYIIZIII

    - 0.0003518893528389501 * XZZXIIZIII

    - 0.42448175317271336 * IIIIIIZIII

    + 0.06732342777645686 * ZIIIIIZIII

    + 0.0782363777898523 * IZIIIIZIII

    + 0.06980180803300681 * IIZIIIZIII

    + 0.07055432072184756 * IIIZIIZIII

    + 0.06878552428444665 * IIIIZIZIII

    + 0.06245512523136934 * IIIIIZZIII

    - 0.0011822832324725804 * IIIYYIIZII

    - 0.0011822832324725804 * IIIXXIIZII

    + 0.0027298828353264134 * YZZZYIIZII

    + 0.0027298828353264134 * XZZZXIIZII

    - 0.0003518893528389501 * YZZYIIIZII

    - 0.0003518893528389501 * XZZXIIIZII

    - 0.42448175317271336 * IIIIIIIZII

    + 0.06732342777645686 * ZIIIIIIZII

    + 0.06980180803300681 * IZIIIIIZII

    + 0.0782363777898523 * IIZIIIIZII

    + 0.07055432072184756 * IIIZIIIZII

    + 0.06878552428444665 * IIIIZIIZII

    + 0.06245512523136934 * IIIIIZIZII

    + 0.06558452315458405 * IIIIIIZZII

    - 0.00154067008970742 * IIIYYIIIZI

    - 0.00154067008970742 * IIIXXIIIZI

    + 0.0023679368995844726 * YZZZYIIIZI

    + 0.0023679368995844726 * XZZZXIIIZI

    - 0.00901204279263803 * YZZYIIIIZI

    - 0.00901204279263803 * XZZXIIIIZI

    - 0.38991776474152134 * IIIIIIIIZI

    + 0.060505605672770954 * ZIIIIIIIZI

    + 0.07055432072184756 * IZIIIIIIZI

    + 0.07055432072184756 * IIZIIIIIZI

    + 0.08470391802239534 * IIIZIIIIZI

    + 0.05665606755281972 * IIIIZIIIZI

    + 0.053929860773588405 * IIIIIZIIZI

    + 0.06022550139954594 * IIIIIIZIZI

    + 0.06022550139954594 * IIIIIIIZZI

    + 0.004360552555030484 * IIIIIYZZYZ

    + 0.004360552555030484 * IIIIIXZZXZ

    + 0.011733623912074194 * IIIYYIIIIZ

    + 0.011733623912074194 * IIIXXIIIIZ

    - 0.03305872858775587 * YZZZYIIIIZ

    - 0.03305872858775587 * XZZZXIIIIZ

    + 0.0127339139792953 * YZZYIIIIIZ

    + 0.0127339139792953 * XZZXIIIIIZ

    - 0.30101532158947975 * IIIIIIIIIZ

    + 0.11433954684977561 * ZIIIIIIIIZ

    + 0.06878552428444665 * IZIIIIIIIZ

    + 0.06878552428444665 * IIZIIIIIIZ

    + 0.05665606755281972 * IIIZIIIIIZ

    + 0.12357087224898464 * IIIIZIIIIZ

    + 0.08360121967246183 * IIIIIZIIIZ

    + 0.06278876343471208 * IIIIIIZIIZ

    + 0.06278876343471208 * IIIIIIIZIZ

    + 0.053621410722614865 * IIIIIIIIZZ
    """
    # paulis = convert_hamiltonian(input_hamiltonian)
    # He = sum(paulis).to_matrix()
    #
    # paulis = [2 * X ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I ^ I, 6.2 * I ^ Y ^ I ^ Y ^ I ^ I ^ I ^ I ^ I ^ I,
    #        4.1 * I ^ I ^ I ^ Y ^ I ^ Z ^ I ^ I ^ I ^ I, 5 * I ^ I ^ I ^ I ^ X ^ Z ^ I ^ I ^ I ^ I,
    #        3.4 * I ^ I ^ I ^ I ^ X ^ I ^ I ^ X ^ I ^ I, 3 * I ^ I ^ I ^ I ^ X ^ Z ^ I ^ I ^ I ^ I,
    #        2.4 * I ^ I ^ I ^ I ^ I ^ Z ^ I ^ X ^ I ^ I, 5 * I ^ X ^ I ^ I ^ X ^ I ^ I ^ I ^ I ^ I,
    #        3.7 * I ^ I ^ X ^ I ^ I ^ I ^ I ^ X ^ I ^ I,
    #        2.4 * I ^ I ^ I ^ X ^ I ^ I ^ I ^ X ^ I ^ I, 3 * I ^ I ^ I ^ I ^ I ^ Z ^ X ^ I ^ I ^ I,
    #        2.3 * I ^ X ^ I ^ I ^ I ^ I ^ I ^ Y ^ I ^ I, 5.2 * I ^ I ^ I ^ I ^ X ^ I ^ Z ^ I ^ I ^ I,
    #        4.7 * I ^ I ^ Z ^ I ^ I ^ I ^ Z ^ I ^ I ^ I,
    #        1 * X ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X, Z ^ I ^ I ^ I ^ I ^ I ^ I ^ I ^ X ^ I]
    # He = sum(paulis).to_matrix()

    from qiskit.quantum_info import Pauli
    from qiskit.opflow.primitive_ops import PauliOp

    local_terms = ['I'*i + 'Z' + 'I'*(num_qubits-1-i) for i in range(num_qubits)]
    twobodyX_terms = ['I'*i + 'XX' + 'I'*(num_qubits-2-i) for i in range(num_qubits-1)]
    twobodyX_terms.append('X' + 'I'*(num_qubits-2) + 'X')
    twobodyY_terms = ['I'*i + 'YY' + 'I'*(num_qubits-2-i) for i in range(num_qubits-1)]
    twobodyY_terms.append('Y' + 'I'*(num_qubits-2) + 'Y')
    paulis = []
    paulis += [PauliOp(Pauli(term), coeff=0.75) for term in local_terms]
    paulis += [PauliOp(Pauli(term), coeff=1.) for term in twobodyX_terms]
    paulis += [PauliOp(Pauli(term), coeff=1.) for term in twobodyY_terms]
    He = sum(paulis).to_matrix()

    trotterizer = Trotterizer(num_qubits, paulis, 'ZX_cycle')
    trotter_circuit = trotterizer.trotterize(num_layers=1)


    from qiskit.algorithms.optimizers import ADAM
    from qiskit.quantum_info import Statevector
    from scipy.optimize import minimize
    x0 = np.random.normal(size=trotter_circuit.num_parameters)
    print(trotter_circuit.num_parameters)
    print(trotter_circuit.parameters)

    def func(x):
        init_state = Statevector.from_label('0'*num_qubits)
        final_state = init_state.evolve(trotter_circuit.assign_parameters(dict(zip(trotter_circuit.parameters, x)))).data
        energy = np.real(final_state.conj().T @ He @ final_state)
        print(energy)
        return energy

    res = minimize(func, x0, method='L-BFGS-B')
    print(trotter_circuit.num_parameters)
    print(res)

    print(min(np.linalg.eigvals(He)))

    ## 1 layer  -1.081405994084623
    ## 2 layers -1.0814059812452388
