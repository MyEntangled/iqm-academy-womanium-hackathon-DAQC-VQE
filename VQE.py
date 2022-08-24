import numpy as np

from qiskit.quantum_info import Statevector

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, SLSQP
#from qiskit.algorithms.optimizers import *
from qiskit.circuit.library import TwoLocal

from qiskit import transpile

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import *
#from qiskit import IBMQ

from qiskit.opflow.expectations import PauliExpectation
from qiskit.opflow.converters import CircuitSampler
from qiskit.opflow.state_fns import StateFn, CircuitStateFn

import helper
from Trotterizer import Trotterizer
from SwapMap import SwapMap

from scipy.optimize import minimize

from qiskit.providers.aer import AerSimulator
from qiskit.opflow import I,X,Y,Z

from qiskit.quantum_info import Pauli
from qiskit.opflow.primitive_ops import PauliOp


seed = 170
algorithm_globals.random_seed = seed

class VQE_Solver():
    def __init__(self, paulis):
        self.paulis = paulis
        self.pauli_strings = [pauli.primitive.__str__() for pauli in self.paulis]
        self.H_prob = sum(paulis)
        self.num_qubits = len(self.pauli_strings[0])

    def transform_paulis(self, qubit_order, update=True):

        transformed_pauli_strings = [''.join([pauli_string[qubit] for qubit in qubit_order]) for pauli_string in
                                     self.pauli_strings]
        transformed_paulis = [PauliOp(Pauli(transformed_pauli_strings[idx]), coeff=self.paulis[idx].coeff) for idx in
                              range(len(self.paulis))]

        if update is True:
            self.paulis = transformed_paulis
            self.pauli_strings = transformed_pauli_strings
            self.H_prob = sum(self.paulis)
            return self.paulis
        else:
            return transformed_paulis

    def true_ground_state_energy(self):
        npme = NumPyMinimumEigensolver()
        result = npme.compute_minimum_eigenvalue(operator=self.H_prob)
        ref_value = result.eigenvalue.real
        return ref_value


    def numerical_solve(self, ansatz):
        He = self.H_prob.to_matrix()

        x0 = np.random.normal(size=ansatz.num_parameters)
        init_state = Statevector.from_label('0' * self.num_qubits)
        def func(x):
            final_state = init_state.evolve(
                ansatz.assign_parameters(dict(zip(ansatz.parameters, x)))).data
            energy = np.real(final_state.conj().T @ He @ final_state)
            print(energy)
            return energy

        res = minimize(func, x0, method='SLSQP', options={'maxiter': 200})
        return res

    def statevector_solve(self, ansatz):
        # define your backend or quantum instance (where you can add settings)
        seed = 170
        algorithm_globals.random_seed = seed

        #qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
        qi = QuantumInstance(backend=Aer.get_backend('statevector_simulator'))

        psi = CircuitStateFn(ansatz)
        measurable_expression = StateFn(self.H_prob, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)

        def func(x):
            sampler = CircuitSampler(qi, statevector=True).convert(expectation, dict(zip(ansatz.parameters, x)))
            energy = sampler.eval().real
            print(energy)
            return energy

        x0 = np.random.normal(size=ansatz.num_parameters)
        res = minimize(func, x0, method='SLSQP')
        return res

    def noiseless_simulator_solve(self, ansatz):
        # define your backend or quantum instance (where you can add settings)
        backend = Aer.get_backend('aer_simulator')
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                             shots=4096)

        psi = CircuitStateFn(ansatz)
        measurable_expression = StateFn(self.H_prob, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)

        def func(x):
            sampler = CircuitSampler(qi).convert(expectation, dict(zip(ansatz.parameters, x)))
            energy = sampler.eval().real
            print(energy)
            return energy

        x0 = np.random.normal(size=ansatz.num_parameters)
        optimizer = SLSQP()
        res = optimizer.minimize(fun=func,x0=x0)
        return res

    def noisy_simulator_solve(self, ansatz):
        # provider = IBMQ.load_account()
        # device_backend = provider.get_backend('ibmq_manila')
        # device = QasmSimulator.from_backend(device_backend)
        # coupling_map = device.configuration().coupling_map
        # noise_model = NoiseModel.from_backend(device)
        # basis_gates = noise_model.basis_gates

        device_backend = FakeVigo()
        backend = Aer.get_backend('aer_simulator')
        noise_model = None
        device = QasmSimulator.from_backend(device_backend)
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates

        # print(noise_model)
        # print()

        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                             coupling_map=coupling_map, noise_model=noise_model, )

        # transpiled_ansatz = transpile(ansatz, basis_gates=basis_gates, optimization_level=3)
        # print(transpiled_ansatz.draw())

        psi = CircuitStateFn(ansatz)
        measurable_expression = StateFn(self.H_prob, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)

        def func(x):
            circuit_sampler = CircuitSampler(qi)
            bound_expectation = expectation.assign_parameters(dict(zip(expectation.parameters, x)))

            sampler = circuit_sampler.convert(bound_expectation)
            energy = sampler.eval().real
            print(energy)
            return energy

        x0 = np.random.normal(size=ansatz.num_parameters)
        optimizer = SPSA()
        res = optimizer.minimize(fun=func,x0=x0)

        # spsa = SPSA(maxiter=10)
        # var_form = TwoLocal(num_qubits=len(self.pauli_strings[0]),rotation_blocks='ry', entanglement_blocks='cz')
        # print(ansatz.draw())
        # print('-------')
        # print(transpiled_ansatz.draw())
        # vqe = VQE(var_form, optimizer=spsa, callback=None, quantum_instance=qi)
        # print('VQE start')
        # res = vqe.compute_minimum_eigenvalue(operator=self.H_prob)
        return res


if __name__ == '__main__':
    num_qubits = 5

    ## Define the Hamiltonian by a list of Pauli terms
    paulis = helper.generate_ZZX_chain_hamiltonian(num_qubits, d=0, l=0.75)
    # paulis = [4 * Y ^ I ^ I ^ Y ^ I, 2 * Y ^ Z ^ I ^ I ^ I, 4 * Y ^ I ^ Y ^ I ^ I, 5 * I ^ Y ^ I ^ Y ^ I, 1 * I ^ Z ^ Y ^ I ^ I,
    #  4 * I ^ Y ^ I ^ I ^ X, 2 * I ^ I ^ X ^ I ^ X, I ^ I ^ I ^ Y ^ Z, 4 * Y ^ I ^ I ^ I ^ Z]
    #paulis = helper.generate_random_hamiltonian(num_qubits=num_qubits, interaction_size=2, seed=2002)

    pauli_strings = [pauli.primitive.__str__() for pauli in paulis]
    print(paulis)

    qubit_order, swap_map = SwapMap(paulis).reorder_qubits(map_type='circular')
    print(qubit_order)
    print(swap_map)

    VQE_solver = VQE_Solver(paulis)

    ## Reorder qubits
    VQE_solver.transform_paulis(qubit_order=qubit_order, update=True)
    print(paulis)
    print(VQE_solver.paulis)

    trotterizer = Trotterizer(num_qubits, VQE_solver.paulis, 'ZX_cycle')
    ansatz = trotterizer.multiorder_trotterize(num_layers=1)
    #ansatz = trotterizer.trotterize(num_layers=3)
    #ansatz = TwoLocal(num_qubits=num_qubits, reps=3, rotation_blocks='ry', entanglement_blocks='cz')
    print(ansatz.draw())
    print("Number of parameters", ansatz.num_parameters)

    ground_energy = VQE_solver.true_ground_state_energy()
    print("Ground Energy", ground_energy)

    res = VQE_solver.numerical_solve(ansatz)
    print("Optimization result", res)

