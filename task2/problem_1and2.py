#!/usr/bin/env python

from numpy import array, zeros, zeros_like, dot, diag, diagonal, exp, sign, isclose, pi, linspace, sin, sqrt
from numpy.linalg import norm
from numpy.random import rand, seed
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lobpcg
from matplotlib import pyplot as plt

seed(1108)

# 1
def is_next_neighbor_hopping(num_sites, state1, state2):
	if state1 == state2:
		return 0
	diff = state1 ^ state2
	if state1 & diff == 0 or state2 & diff == 0:
		return 0
	return int(diff % 3 == 0 and diff // 3 & (diff // 3 - 1) == 0) + int(diff == (1 << num_sites - 1) + 1)

def get_spin_value(num_sites, state, index):
	return 1/2 - (state >> index % num_sites & 1)

def xxx_hamiltonian_element(num_sites, state1, state2):
	result = 0
	if state1 == state2:
		for i in range(num_sites):
			result += 1/4 - get_spin_value(num_sites, state1, i) * get_spin_value(num_sites, state1, i + 1)
	result -= is_next_neighbor_hopping(num_sites, state1, state2) / 2
	return result

def full_xxx_hamiltonian(num_sites):
	return [[xxx_hamiltonian_element(num_sites, i, j) for j in range(1 << num_sites)] for i in range(1 << num_sites)]

def sparse_xxx_hamiltonian(num_sites):
	hamiltonian_matrix = csc_matrix((1 << num_sites, 1 << num_sites))
	for state in range(1 << num_sites):
		if state == 0:
			continue
		if state >> num_sites - 1 == 1 and state & 1 == 0:
			neighbor_state = state ^ (1 << num_sites - 1) ^ 1
			hamiltonian_matrix[state, neighbor_state] -= 1/2
			hamiltonian_matrix[neighbor_state, state] -= 1/2
		adjusted_state = state & -2
		while adjusted_state > 0:
			new_adjusted_state = adjusted_state & (adjusted_state - 1)
			neighbor_down_state = (adjusted_state ^ new_adjusted_state) >> 1
			if state & neighbor_down_state == 0:
				neighbor_state = state ^ (neighbor_down_state * 3)
				hamiltonian_matrix[state, neighbor_state] -= 1/2
				hamiltonian_matrix[neighbor_state, state] -= 1/2
			adjusted_state = new_adjusted_state
		for i in range(num_sites):
			hamiltonian_matrix[state, state] += 1/4 - get_spin_value(num_sites, state, i) * get_spin_value(num_sites, state, i + 1)
	return hamiltonian_matrix

# 2
def qr_decomposition(A):
	rows, cols = A.shape
	Q = zeros((rows, cols))
	R = zeros((cols, cols))
	for i in range(cols):
		v = A[:, i]
		for j in range(i):
			R[j, i] = dot(Q[:, j], A[:, i])
			v -= R[j, i] * Q[:, j]
		R[i, i] = norm(v)
		Q[:, i] = v / R[i, i]
	return Q, R

def qr_eigenvalue_decomposition(H, tolerance=1e-10, max_iterations=1000):
	H_k = array(H)
	for _ in range(max_iterations):
		Q, R = qr_decomposition(H_k)
		H_k = R @ Q
		if norm(H_k - diag(diagonal(H_k))) < tolerance:
			break
	return H_k