import numpy as np
import pylab

from qiskit.quantum_info import Statevector

from qiskit import Aer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
#from qiskit.algorithms.optimizers import SPSA, ADAM, SciPyOptimizer, SLSQP, L_BFGS_B, COBYLA
from qiskit.algorithms.optimizers import *
from qiskit.circuit.library import TwoLocal

from qiskit import transpile

from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.fake_provider import *
from qiskit import IBMQ

from qiskit.opflow.expectations import ExpectationFactory, PauliExpectation
from qiskit.opflow.converters import CircuitSampler
from qiskit.opflow.state_fns import StateFn, CircuitStateFn

import Connectivity
import helper
from Trotterizer import Trotterizer
from SwapMap import SwapMap

from scipy.optimize import minimize

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

seed = 170
algorithm_globals.random_seed = seed

#IBMQ.delete_accounts()
#IBMQ.save_account('336cb0a05aa0aa2b89f35d3ebc7b1ec6fe2626d035b0722df5c3a9771c3c889ac55407473ed423b6f0f0636b78a156999ad7bfa070121a0041620d5575a618ce')
#IBMQ.update_account('336cb0a05aa0aa2b89f35d3ebc7b1ec6fe2626d035b0722df5c3a9771c3c889ac55407473ed423b6f0f0636b78a156999ad7bfa070121a0041620d5575a618ce')
class VQE_Solver():
    def __init__(self, paulis, ansatz):
        self.paulis = paulis
        self.pauli_strings = [pauli.primitive.__str__() for pauli in self.paulis]

        self.H_prob = sum(paulis)
        self.ansatz = ansatz

    def _transform_paulis(self, transforms):
        transformed_pauli_strings = []
        for i in range(len(self.pauli_strings)):
            string = self.pauli_strings[i]
            for transform in transforms:
                string = string.replace(transform[0], transform[1])
            transformed_pauli_strings.append(string)

        transformed_paulis = [PauliOp(Pauli(transformed_pauli_strings[idx]), coeff=self.paulis[idx].coeff)
                              for idx in range(len(self.paulis))]
        return transformed_paulis

    def true_ground_state_energy(self, observable_transform=None):
        npme = NumPyMinimumEigensolver()
        if observable_transform:
            result = npme.compute_minimum_eigenvalue(operator=sum(self._transform_paulis(observable_transform)))
        else:
            result = npme.compute_minimum_eigenvalue(operator=self.H_prob)
        ref_value = result.eigenvalue.real
        return ref_value


    def numerical_solve(self, observable_transform=None):
        if observable_transform:
            transformed_paulis = self._transform_paulis(observable_transform)
            He = sum(transformed_paulis).to_matrix()
        else:
            He = self.H_prob.to_matrix()

        x0 = np.random.normal(size=self.ansatz.num_parameters)
        init_state = Statevector.from_label('0' * num_qubits)
        def func(x):
            final_state = init_state.evolve(
                self.ansatz.assign_parameters(dict(zip(self.ansatz.parameters, x)))).data
            energy = np.real(final_state.conj().T @ He @ final_state)
            print(energy)
            return energy

        res = minimize(func, x0, method='SLSQP', options={'maxiter': 200})
        return res

    def statevector_solve(self):
        # define your backend or quantum instance (where you can add settings)
        seed = 170
        algorithm_globals.random_seed = seed

        #qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed)
        qi = QuantumInstance(backend=Aer.get_backend('statevector_simulator'))

        psi = CircuitStateFn(self.ansatz)
        measurable_expression = StateFn(self.H_prob, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)

        def func(x):
            sampler = CircuitSampler(qi, statevector=True).convert(expectation, dict(zip(self.ansatz.parameters, x)))
            energy = sampler.eval().real
            print(energy)
            return energy

        x0 = np.random.normal(size=self.ansatz.num_parameters)
        res = minimize(func, x0, method='SLSQP')
        return res

    def noiseless_simulator_solve(self):
        # define your backend or quantum instance (where you can add settings)
        backend = Aer.get_backend('aer_simulator')
        qi = QuantumInstance(backend=backend, seed_simulator=seed, seed_transpiler=seed,
                             shots=4096)

        psi = CircuitStateFn(self.ansatz)
        measurable_expression = StateFn(self.H_prob, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)

        def func(x):
            sampler = CircuitSampler(qi).convert(expectation, dict(zip(self.ansatz.parameters, x)))
            energy = sampler.eval().real
            print(energy)
            return energy

        x0 = np.random.normal(size=self.ansatz.num_parameters)
        optimizer = SPSA()
        res = optimizer.minimize(fun=func,x0=x0)
        return res

    def noisy_simulator_solve(self):
        os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'
        #device_backend = FakeNairobi()

        provider = IBMQ.load_account()
        device_backend = provider.get_backend('ibmq_manila')
        device = QasmSimulator.from_backend(device_backend)
        coupling_map = device.configuration().coupling_map
        noise_model = NoiseModel.from_backend(device)
        basis_gates = noise_model.basis_gates

        print(noise_model)
        print()

        qi = QuantumInstance(backend=Aer.get_backend('aer_simulator'), seed_simulator=seed, seed_transpiler=seed,
                             coupling_map=coupling_map, noise_model=noise_model)

        transpiled_ansatz = transpile(self.ansatz, basis_gates=basis_gates, optimization_level=3)
        print(transpiled_ansatz.draw())
        psi = CircuitStateFn(transpiled_ansatz)
        measurable_expression = StateFn(self.H_prob, is_measurement=True).compose(psi)
        expectation = PauliExpectation().convert(measurable_expression)

        def func(x):
            print(x)
            circuit_sampler = CircuitSampler(qi)
            print('CIRCUIT SAMPLER')
            bound_expectation = expectation.assign_parameters(dict(zip(expectation.parameters, x)))
            print(bound_expectation)
            sampler = circuit_sampler.convert(bound_expectation)
            print('SAMPLER')
            energy = sampler.eval().real
            print(energy)
            return energy

        x0 = np.random.normal(size=self.ansatz.num_parameters)
        optimizer = SPSA()
        print('HI')
        res = optimizer.minimize(fun=func,x0=x0)
        return res


    def plot_training(self, counts, values, min_value):
        pylab.rcParams['figure.figsize'] = (12, 4)
        pylab.plot(counts, values)
        pylab.hlines(y=min_value, color='r', linestyle='-')
        pylab.xlabel('Eval count')
        pylab.ylabel('Energy')
        pylab.title('Convergence with no noise')
        pylab.grid()
        pylab.plot()

if __name__ == '__main__':
    num_qubits = 6

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
    # Z: -13.267
    # Y: -13.249
    # X: -13.279

    paulis = helper.generate_random_hamiltonian(num_qubits=num_qubits, interaction_size=3, seed=2711)
    print(paulis)

    swap_map = SwapMap(paulis).get_swap_map(map_type='circular')
    print(swap_map)
    trotterizer = Trotterizer(num_qubits, paulis, 'ZX_cycle')
    ansatz = trotterizer.trotterize(num_layers=2, swap_map=swap_map)
    print(ansatz.num_parameters)

    # ansatz = TwoLocal(num_qubits, rotation_blocks='ry', entanglement_blocks='cz',
    #                   entanglement='circular', reps=5)
    # print(ansatz.num_parameters)

    VQE_solver = VQE_Solver(paulis, ansatz)

    transform = None
    transform = [('X', 'Z'), ('Y', 'X'), ('Z', 'Y')]
    transform = [('X', 'Y'), ('Y', 'Z'), ('Z', 'X')]

    ground_energy = VQE_solver.true_ground_state_energy(observable_transform=transform)
    print("Ground Energy", ground_energy)
    # print('Difference', noiseless_energy - ground_energy)


    res = VQE_solver.numerical_solve(observable_transform=transform)
    print("Optimized function value", res.fun)
    print(res.message, res.success)

