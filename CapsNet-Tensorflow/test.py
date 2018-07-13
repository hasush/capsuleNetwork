import numpy as np
from scipy.io.matlab import loadmat
import matplotlib.pyplot as plt

path = '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/data/affNist/test.mat'

mat_dict = loadmat(path)
data = mat_dict['affNISTdata']

# Remove redundant arrays surrounding the data.
data = data[0][0]
assert len(data) == 8

# Index 2 contains the images.
# An image is given by data[2][:,sample]
# because data[2] has shape (1600, 320000)
# A Label is given by data[5][:,sample]
# A one hot label is given by data[4][:,sample]

# Visualize the contents of the data.
for i in range(8):
	print(i, len(data[i]))
	print(i, data[i])

print(len(data[0][0]))
digit_sample = 224
sample_label = data[5][0][digit_sample]
sample_onehot = data[4][:,digit_sample]

print('sample label: ', sample_label)
print('sample onehot: ', sample_onehot)
sample = data[2][:,digit_sample]
print(sample.shape)
sample = np.reshape(sample, (40,40))
plt.imshow(sample)
plt.show()
# print(len(sample))

# print(len(data[0][0][0]))
# print(data[0])


