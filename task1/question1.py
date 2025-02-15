#!/usr/bin/env python

from time import time

from numpy import hstack, vstack, zeros
from numpy.random import rand, seed
import matplotlib.pyplot as plt

seed(1108)

# a
def split_submatrices(matrix):
	mid_point = matrix.shape[0] // 2
	return matrix[:mid_point, :mid_point], matrix[:mid_point, mid_point:], matrix[mid_point:, :mid_point], matrix[mid_point:, mid_point:]

def merge_submatrices(sub_11, sub_12, sub_21, sub_22):
	top_part = hstack((sub_11, sub_12))
	bottom_part = hstack((sub_21, sub_22))
	return vstack((top_part, bottom_part))

def recursive_multiply(matrix_A, matrix_B):
	size = matrix_A.shape[0]
	if size == 1:
		return matrix_A * matrix_B
	A11, A12, A21, A22 = split_submatrices(matrix_A)
	B11, B12, B21, B22 = split_submatrices(matrix_B)
	C11 = recursive_multiply(A11, B11) + recursive_multiply(A12, B21)
	C12 = recursive_multiply(A11, B12) + recursive_multiply(A12, B22)
	C21 = recursive_multiply(A21, B11) + recursive_multiply(A22, B21)
	C22 = recursive_multiply(A21, B12) + recursive_multiply(A22, B22)
	return merge_submatrices(C11, C12, C21, C22)

def pad_matrix_power_of_two(matrix):
	num_rows, num_cols = matrix.shape
	power_of_two = 1 << (max(num_rows, num_cols) - 1).bit_length()
	padded = zeros((power_of_two, power_of_two), dtype=matrix.dtype)
	padded[:num_rows, :num_cols] = matrix
	return padded

def matrix_product(matrix_A, matrix_B):
	size = matrix_A.shape[0]
	matrix_A = pad_matrix_power_of_two(matrix_A)
	matrix_B = pad_matrix_power_of_two(matrix_B)
	return recursive_multiply(matrix_A, matrix_B)[:size, :size]

# b
# T(n) = a T(n/b) + f(n)
# a = 8, b = 2, f(n) = O(n^2)
# c_crit = log(a) / log(b) = 3
# T(n) = O(n^3)

# c
def complexity_estimate(size):
	return size**3

def benchmark(matrix_size, trials=1):
	elapsed_time = 0
	for _ in range(trials):
		matrix_A = rand(matrix_size, matrix_size)
		matrix_B = rand(matrix_size, matrix_size)
		start = time()
		matrix_product(matrix_A, matrix_B)
		elapsed_time += time() - start
	return elapsed_time / trials / complexity_estimate(matrix_size)

test_sizes = [2**i for i in range(4, 9)]
test_results = [benchmark(size) for size in test_sizes]
plt.plot(test_sizes, test_results)
plt.xlabel('Matrix size')
plt.ylabel('Time / Complexity estimate')
plt.show()
# They largely agree, because the curve is nearly flat.
