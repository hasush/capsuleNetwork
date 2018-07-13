import matplotlib.pyplot as plt

capsule_error = [0.9373, 0.9652, 0.9778, 0.9858, 0.9899, 0.9934, 0.9942, 0.9944]
cnn_error = [0.9224, 0.9619, 0.9756, 0.9833, 0.9900, 0.9923, 0.9941, 0.9933]

training_set_size = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

assert len(training_set_size) == len(capsule_error)
assert len(training_set_size) == len(cnn_error)

plt.figure()
plt.plot(training_set_size, capsule_error, 'r', label='capsule')
plt.plot(training_set_size, cnn_error, 'b', label='cnn')
plt.xlabel('Percentage of training data')
plt.ylabel('Accuracy')
plt.title('Accuracy on Test When Varying Percentage of Training Data')
plt.legend()
plt.show()
