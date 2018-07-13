import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##################
### FILE PATHS ###
##################

filePathDicts = ['/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/cnn_percent_data_epoch_constant_test/cnn_percent_data_10_epoch_constant/results']
				 # '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/cnn_percent_data_10_epoch_constant/results',
				 # '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/cnn_percent_data_25_epoch_constant/results',
				 # '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/cnn_percent_data_50_epoch_constant/results',
				 # '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/cnn_percent_data_75_epoch_constant/results',
				 # '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/cnn_percent_data_100/results']

legendEntriesLoss = ['1%']#,'10%', '25%', '50%', '75%', '100%']
legendEntriesAccuracy = ['Train: 1%']#, 'Train: 10%','Val: 10%', 'Train: 25%', 'Val: 25%', 'Train: 50%', 'Val: 50%', 'Train: 75%', 'Val: 75%', 'Train: 100%', 'Val: 100%']
legendEntriesAccuracy = ['Val: 1%']#, 'Val: 10%', 'Val: 25%', 'Val: 50%', 'Val: 75%', 'Val: 100%']

assert len(legendEntriesLoss) == len(filePathDicts)
assert len(legendEntriesAccuracy) == len(filePathDicts)

lossVec = []
trainAccVec = []
valAccVec = []

for index, filePathDict in enumerate(filePathDicts):

	# Get the path to the individual files.
	lossFilePath = os.path.normpath(filePathDict + '/loss.csv')
	trainAccFilePath = os.path.normpath(filePathDict + '/train_acc.csv')
	valAccFilePath = os.path.normpath(filePathDict + '/val_acc.csv')

	# Read in the csv files.
	loss = pd.read_csv(lossFilePath, sep=',')
	trainAcc = pd.read_csv(trainAccFilePath, sep=',')
	valAcc = pd.read_csv(valAccFilePath, sep=',')

	# Set the index.
	loss = loss.set_index('step')
	trainAcc = trainAcc.set_index('step')
	valAcc = valAcc.set_index('step')

	# Plot the loss.
	if index == 0:
		ax_loss = loss.plot()
	else:
		loss.plot(ax=ax_loss)

	# Plot the accuracy.
	if index == 0:
		# ax_acc = trainAcc.plot()
		ax_acc = valAcc.plot()
	else:
		valAcc.plot(ax=ax_acc)
	# valAcc.plot(ax=ax_acc)

# Plot labels, etc.
ax_loss.set_xlabel('Steps')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss for Training Set')
ax_loss.legend(legendEntriesLoss)

ax_acc.set_xlabel('Steps')
ax_acc.set_ylabel('Accuracy')
ax_acc.set_title("Accuracy for Validation Sets")
ax_acc.legend(legendEntriesAccuracy)
plt.show()