import numpy as np
import matplotlib.pyplot as plt



def reshape_testing():

	# Reshape testing.
	np.random.seed(0)
	test = np.random.random((5, 6, 4))
	test1 = np.reshape(test, (5, 3, 2, 4))
	test2 = np.reshape(test, (5, 2, 3, 4))

	print('test\n',test)
	print('test1\n',test1)
	print('test2\n',test2)

	matrix1 = np.array([[1,2,3],[4,5,6]])
	matrix2 = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

	print('matrix1:\n', matrix1, '\n')
	print('matrix2:\n', matrix2, '\n')

	tiled_matrix1 = np.reshape(np.tile(matrix1, (1, matrix2.shape[1])), (matrix2.shape[1]*matrix1.shape[0], matrix1.shape[1]))
	tiled_matrix2 = np.tile(matrix2.T, (matrix1.shape[0],1))

	print('tiled1:\n', tiled_matrix1, '\n')
	print('tiled2:\n', tiled_matrix2, '\n')

	result_matmul = np.matmul(matrix1, matrix2)
	result_tiled = np.reshape(np.sum(tiled_matrix1*tiled_matrix2, axis=1), (matrix1.shape[0], matrix2.shape[1]))

	print('result matmul:\n', result_matmul, '\n')
	print('result tiled:\n', result_tiled, '\n')

def tiled_matrix_multiplication(matrix1, matrix2):
	""" Tiled matrix multiplication of 2D matrices.
		result = matrix1 * matrix2 :: a x c = a x b * b x c  """

	# Check 2D Matrices.
	assert len(matrix1.shape) == 2
	assert len(matrix2.shape) == 2

	# Check dimensions match.
	assert matrix1.shape[1] == matrix2.shape[0]

	# Create tiled representations for input into pointwise multiplication, reduce summing, and reshaping.
	tiled_matrix1 = np.reshape(np.tile(matrix1, (1, matrix2.shape[1])), (matrix2.shape[1]*matrix1.shape[0], matrix1.shape[1]))
	tiled_matrix2 = np.tile(matrix2.T, (matrix1.shape[0], 1))

	# Compute the matrix product of the tiled representations.
	result_tiled = np.reshape(np.sum(tiled_matrix1*tiled_matrix2, axis=1), (matrix1.shape[0], matrix2.shape[1]))

	return result_tiled 

def load_arrays():
	labels = np.load('saved_arrays/labels.npy')
	v_j = np.load('saved_arrays/v_j.npy')
	weight_matrix = np.load('saved_arrays/weight_matrix.npy')
	b_ij = np.load('saved_arrays/b_ij.npy')
	s_j = np.load('saved_arrays/s_j.npy')
	c_ij = np.load('saved_arrays/c_ij.npy')
	u_hat = np.load('saved_arrays/u_hat.npy')
	biases = np.load('saved_arrays/biases.npy')

	return labels, v_j, weight_matrix, b_ij, s_j, c_ij, u_hat, biases

def main():
	labels, v_j, weight_matrix, b_ij, s_j, c_ij, u_hat, biases = load_arrays()

	for index in range(1):
		print(c_ij.shape)
		c_ij_sample = np.squeeze(c_ij[index])
		print(c_ij_sample.shape)
		asdf = np.zeros((1152,1000))
		for i in range(1152):
			for j in range(10):
				for k in range(100):
					asdf[i][j*100 + k] = c_ij_sample[i][j]
		label = labels[index]
		print('Label: ', label)
		plt.imshow(asdf, cmap='gray')

		# plt.imshow(c_ij_sample, cmap='gray')
		# plt.axis('equal')
		# plt.show()

	# c_ij_sample = np.squeeze(c_ij[0])

	print('matrix shape before reshape:', weight_matrix.shape)

	weight_matrix = np.squeeze(np.reshape(weight_matrix, [1, 1152, 10, 16, 8, 1]))
	print('matrix shape after reshape and squeeze: ', weight_matrix.shape)
	weight_matrix = weight_matrix[327][9]

	print('max value: ', np.max(weight_matrix))
	print('min value: ', np.min(weight_matrix))

	column = weight_matrix[:,3]
	reshaped_column = np.reshape(column, (4,4))
	plt.matshow(reshaped_column)

	# plt.imshow(c_ij_sample)
	plt.matshow(weight_matrix)
	print(weight_matrix)
	plt.show()




	# c_ij_sample = np.reshape(np.squeeze(c_ij[0]), (36, 32, 10))

	# print(c_ij_sample.shape)

	# c_ij_sample_digit = c_ij_sample
	# print(c_ij_sample_digit.shape)
	# print(c_ij_sample_digit)
	# print(np.max(c_ij_sample_digit))

	# digit_of_interest = 9
	# c_ij_sample_digit = np.zeros((36, 32))

	# for i in range(36):
	# 	for j in range(32):
	# 		c_ij_sample_digit[i][j] = c_ij_sample[i][j][digit_of_interest]

	# print(c_ij_sample_digit)
	# plt.matshow(c_ij_sample_digit)
	# plt.show()



if __name__ == "__main__":
	main()







# print('matrix shape before reshape:', weight_matrix.shape)

# weight_matrix = np.squeeze(np.reshape(weight_matrix, [1, 1152, 10, 16, 8, 1]))
# print('matrix shape after reshape and squeeze: ', weight_matrix.shape)

# print('max value: ', np.max(weight_matrix))


# while True:
# 	i = np.random.randint(1152)
# 	j = np.random.randint(10)
# 	example = weight_matrix[i][j]

# 	plt.matshow(example)

# 	# plt.figure()
# 	# plt.hist(weight_matrix.flatten())
# 	plt.show()

