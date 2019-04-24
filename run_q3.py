import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 16
learning_rate = 0.002
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,train_y.shape[1],params,'output')


training_data = np.zeros((max_iters, 2))
validation_data = np.zeros((max_iters, 2))
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, "layer1", sigmoid)
        probs = forward(h1, params, 'output', softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc
        # backward
        delta = probs - yb
        delta = backwards(delta, params, "output", linear_deriv)
        delta = backwards(delta, params, "layer1", sigmoid_deriv)

        # apply gradient 
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']
        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']

    total_loss /= train_x.shape[0]
    total_acc /= batch_num 
        
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))
        

    h1 = forward(valid_x, params, "layer1", sigmoid)
    probs = forward(h1, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
    valid_loss /= valid_x.shape[0]

    training_data[itr, :] = np.asarray([total_loss, total_acc])
    validation_data[itr, :] = np.asarray([valid_loss, valid_acc])

# Plot training results
if False:
    ax1 = plt.subplot(121)
    ax1.plot(training_data[:,0], 'g', label='Training loss')
    ax1.plot(validation_data[:,0], 'b', label="Validation loss")
    ax1.grid(True)
    ax1.legend()
    ax1.set_title("Loss vs Epochs")

    ax2 = plt.subplot(122)
    ax2.plot(training_data[:,1], 'g', label='Training accuracy')
    ax2.plot(validation_data[:,1], 'b', label="Validation accuracy")
    ax2.grid(True)
    ax2.set_ylim((0, 1))
    ax2.legend()
    ax2.set_title("Accuracy vs Epochs")

    plt.show()

# run on validation set and report accuracy! should be above 75%
h1 = forward(valid_x, params, "layer1", sigmoid)
probs = forward(h1, params, 'output', softmax)
valid_loss, valid_acc = compute_loss_and_acc(valid_y, probs)
valid_loss /= valid_x.shape[0]

print('Validation accuracy: ',valid_acc)
if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

if False:
    # fig = plt.figure(1)
    # weights = params['Wlayer1']
    # n = weights.shape[-1]

    # grid = ImageGrid(fig, 111, (8, 8))
    # for i in range(n):
    #     grid[i].imshow(weights[:, i].reshape(32, 32))
    # plt.show()

    fig = plt.figure(1)
    initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
    weights_o = params['Wlayer1']

    ax2 = plt.subplot(122)
    grid = ImageGrid(fig, 111, (8, 8))
    for i in range(64):
        grid[i].imshow(weights_o[:, i].reshape(32, 32))
    plt.show()



# Q3.1.3
test_data = scipy.io.loadmat('../data/nist36_test.mat')
test_x, test_y = test_data['test_data'], test_data['test_labels']
h1 = forward(test_x, params, 'layer1')
probs = forward(h1, params, 'output', softmax)
labels = np.argmax(test_y, axis=1)
preds = np.argmax(probs, axis=1)

confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
for label, pred in zip(labels, preds):
    confusion_matrix[label, pred] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()