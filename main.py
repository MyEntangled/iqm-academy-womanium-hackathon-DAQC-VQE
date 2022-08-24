from qiskit.opflow import I,X,Y,Z

from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp

from VQE import *
import helper

from typing import Union

def DAVQE(num_qubits:int, paulis:Union[str,list], davqe_ansatz:bool, num_layers:int):
    if paulis == 'ZZX-chain':
        paulis = helper.generate_ZZX_chain_hamiltonian(num_qubits, d=0, l=0.75)
    elif paulis == 'random':
        paulis = helper.generate_random_hamiltonian(num_qubits=num_qubits, interaction_size=2, seed=2711)
    elif paulis == 'H2':
        paulis = helper.generate_H2_hamiltonian()
    elif paulis == 'LiH':
        paulis = helper.generate_LiH_hamiltonian()
    elif paulis == 'asym':
        paulis = [4 * Y ^ I ^ I ^ Y ^ I, 2 * Y ^ Z ^ I ^ I ^ I, 4 * Y ^ I ^ Y ^ I ^ I, 5 * I ^ Y ^ I ^ Y ^ I, 1 * I ^ Z ^ Y ^ I ^ I,
        4 * I ^ Y ^ I ^ I ^ X, 2 * I ^ I ^ X ^ I ^ X, I ^ I ^ I ^ Y ^ Z, 4 * Y ^ I ^ I ^ I ^ Z]
    else:
        assert isinstance(paulis, list)

    pauli_strings = [pauli.primitive.__str__() for pauli in paulis]
    assert len(pauli_strings[0]) == num_qubits

    VQE_solver = VQE_Solver(paulis)
    ground_energy = VQE_solver.true_ground_state_energy()
    print("Ground Energy", ground_energy)

    if davqe_ansatz:
        ## Reorder qubits
        qubit_order, swap_map = SwapMap(paulis).reorder_qubits(map_type='circular')
        VQE_solver.transform_paulis(qubit_order=qubit_order, update=True)

        trotterizer = Trotterizer(num_qubits, VQE_solver.paulis, 'ZX_cycle')
        ansatz = trotterizer.multiorder_trotterize(num_layers=num_layers)

        res = VQE_solver.numerical_solve(ansatz)
        print("Optimization result", res)

    else:
        ansatz = TwoLocal(num_qubits=num_qubits, reps=num_layers, rotation_blocks='ry', entanglement_blocks='cz', entanglement='circular')
        res = VQE_solver.numerical_solve(ansatz)
        print("Optimization result", res)
    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    num_qubits = int(input("Number of qubits "))
    paulis = str(input("Hamiltonian "))
    is_Mansatz = bool(input("Use proposed DAVQE ansatz? "))
    num_layers = int(input("Number of layers "))

    DAVQE(num_qubits, paulis, is_Mansatz, num_layers)