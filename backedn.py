# import numpy as np

# # Importing standard Qiskit libraries
# from qiskit import QuantumCircuit, transpile, Aer, IBMQ
# from qiskit.tools.jupyter import *
# from qiskit.visualization import *
# from ibm_quantum_widgets import *
# from qiskit.providers.aer import QasmSimulator

# # Loading your IBM Quantum account(s)
# provider = IBMQ.load_account()

import numpy as np

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, IBMQ
from qiskit.tools.jupyter import *
from qiskit.visualization import *
from ibm_quantum_widgets import *
from qiskit.providers.aer import QasmSimulator
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator
from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA
from qiskit.algorithms.minimum_eigensolvers import VQE
import pylab
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp
from random import uniform
import numpy
# Loading your IBM Quantum account(s)
provider = IBMQ.load_account()
import scipy.constants as const

H2_op = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)

counters = []
values = []

def store_intermediate_result(eval_count, parameters, mean, std):
    counters.append(eval_count)
    values.append(mean)


numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=PauliSumOp(H2_op))
ref_value = result.eigenvalue.real

device = FakeVigo()
coupling_map = device.configuration().coupling_map

# VARIABLES
iterations =  125
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
spsa = SPSA(maxiter=iterations)
LOWER, HIGHER = 1, 500

noise_model = NoiseModel.from_backend(device)
seed = int(uniform(LOWER, HIGHER))


ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
spsa = SPSA(maxiter = iterations)


noisy_estimator = AerEstimator(
    backend_options={
        "method": "density_matrix",
        "coupling_map": coupling_map,
        "noise_model": noise_model,
    },
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
)

vqe = VQE(
    noisy_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result
)
vqe.estimator = noisy_estimator
result = vqe.compute_minimum_eigenvalue(operator=H2_op)


pylab.rcParams["figure.figsize"] = (12, 4)
pylab.plot(counters, values)
pylab.xlabel("Eval count")
pylab.ylabel("Energy")
pylab.title("Convergence with no noise")

print(seed)

import matplotlib.pyplot as plt

wave_list = []
frequency_list = []
amp = []
x_val = numpy.array(counters)
ex = numpy.fft.fft(x_val)

for i in range(len(values)):
    wavelength = (const.speed_of_light / values[i]) * const.Planck
    wave_list.append(wavelength)
    frequency = (const.speed_of_light / wavelength)  # frequency formula
    frequency_list.append(frequency)

sig_fit = numpy.fft.fft(numpy.sin(frequency_list))
culo = numpy.fft.fft(numpy.sin(counters))
filtered = numpy.fft.ifft(sig_fit.copy())

plt.figure(figsize=(12, 6))
plt.plot(counters, filtered)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# wave_lst = []
# amp_lst = []

# for x in range(len(values)):
#     multiplier = numpy.power(10, 25)
#     #print(multiplier)
#     lbd_num = (const.speed_of_light * const.Planck) #wavelength numerator
#     wave = numpy.abs(lbd_num / values[x]) #wavelength
#     y = numpy.sin((2 * numpy.pi * wave * counters[x]))*multiplier #amplitude
#     wave_lst.append(y)
#     amp_lst.append(y)

# fft = numpy.fft.fft(numpy.sin(wave_lst))
# freq = numpy.fft.fftfreq(fft.shape[-1])
# plt.plot(freq, fft.real, freq, fft.imag)
# plt.show()







