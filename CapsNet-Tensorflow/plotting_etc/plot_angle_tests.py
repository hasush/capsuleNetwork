import os
import sys
import numpy as np
import matplotlib.pyplot as plt



angles = [0,1,2,4,8,16,32,48,64]

capsule_accuracy = [0.994091, 0.994091, 0.993890, 0.993389, 0.991086, 0.972857, 0.806190, 0.510517, 0.268630]
cnn_accracy = [0.991687, 0.991687, 0.990885, 0.991186, 0.987380, 0.970252, 0.816406, 0.5216340, 0.304587]

plt.figure()
plt.plot(angles, capsule_accuracy, 'r', label='capsule')
plt.plot(angles, cnn_accracy, 'b', label='cnn')
plt.legend()
plt.xlabel('Angle in Degrees')
plt.ylabel('Accuracy')
plt.title('Changing Rotation Angle for Testing on MNIST')
plt.show()