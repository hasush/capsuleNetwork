import matplotlib.pyplot as plt
import pandas as pd

### Configurable variables
basepath = '/home/gsandh16/Documents/capsuleNetwork/CapsNet-Tensorflow/archived_results/capsnet_multinist_10/results/'

item_of_interest = 2 # {0:'loss.csv', 1:'train_acc.csv', 2:'val_acc.csv', 3:'test_acc.csv'}
### End configurable variables.

item_dict = {0:'loss.csv', 1:'train_acc.csv', 2:'val_acc.csv', 3:'test_acc.csv'}

filepath = basepath + item_dict[item_of_interest]

data = pd.read_csv(filepath)
data = data.set_index('step')
ax = data.plot(legend=False)
ax.set_xlabel("Steps")
ax.set_ylabel("Accuracy")
ax.set_title("Validation Accuracy Capsnet")
plt.show()
