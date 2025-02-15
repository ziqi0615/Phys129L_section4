#!/usr/bin/env python

from numpy import array, zeros, zeros_like, dot, diag, diagonal, exp, sign, isclose, pi, linspace, sin, sqrt
from numpy.random import seed

from scipy.sparse.linalg import lobpcg
from matplotlib import pyplot as plt
from problem_1and2 import xxx_hamiltonian_element


seed(1108)

def magnon_wavefunction(num_sites, momentum):
	wavefunction = zeros(1 << num_sites, dtype=complex)
	for i in range(num_sites):
		wavefunction[1 << i] = exp(1j * momentum * i)
	return wavefunction

def compute_energy(num_sites, momentum):
	energy_sum = 0
	for i in range(num_sites):
		energy_sum += xxx_hamiltonian_element(num_sites, 1 << 0, 1 << i) * exp(1j * momentum * i)
	return energy_sum.real

num_sites = 30
eigenvalues = [compute_energy(num_sites, 2 * pi * k / num_sites) for k in range(num_sites)]
plt.scatter(range(num_sites), eigenvalues)
k_values = linspace(0, num_sites, 100)
plt.plot(k_values, 2 * sin(pi * k_values / num_sites) ** 2)
plt.xlabel(r'$Np/2\pi$')
plt.ylabel('Energy')
plt.show()
