from __future__ import print_function
import numpy as np
from cs231n.data_utils import load_CIFAR10_mod
import matplotlib.pyplot as plt
from cs231n.classifiers.neural_net import TwoLayerNet
from cs231n.gradient_check import eval_numerical_gradient
from cs231n.vis_utils import visualize_grid
import time

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10_mod(cifar10_dir)
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

# Create a small net and some toy data to check our implementations.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y_lab = np.array([0, 1, 2, 2, 1])
    y = np.zeros((5, 3))
    y[np.arange(5), y_lab] = 1
    return X, y

"""
net = init_toy_model()
X, y = init_toy_data()

scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small.
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))

loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small.
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))

# Use numeric gradient checking to check our implementation of the backward pass.
# If the implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)
print(grads)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
"""

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print(y_test[:10,:])
"""
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (np.sum(net.predict(X_val) * y_val, axis = 1)).mean()
print('Validation accuracy: ', val_acc)

# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Clasification accuracy')
plt.savefig('loss_train_val_acc.png')
plt.show()

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig('nn_vis_grid.png')
    plt.show()

show_net_weights(net) 
"""
#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################

# hidden layer size, learning rate, number of training epochs, and regularization strength.
input_size = 32 * 32 * 3
num_classes = 10
results = {}
best_val = -1
best_net = None # store the best model into this

hidden_sizes = [80, 90, 100]
learning_rates = [1.8e-3, 2e-3, 2.2e-3]
regularization_strengths = [0.5, 0.6, 0.7, 0.8]
batch_sizes = [256, 512, 1024, 2048]

hidden_sizes = [100]
learning_rates = [2e-3]
regularization_strengths = [0.5]
batch_sizes = [1024]

for learning_rate in learning_rates:
    for reg in regularization_strengths:
        for hidden_size in hidden_sizes:
            for batch_size in batch_sizes:
                net = TwoLayerNet(input_size, hidden_size, num_classes)
                stats = net.train(X_train, y_train, X_val, y_val,
                              num_iters=5000, batch_size=batch_size,
                              learning_rate=learning_rate, learning_rate_decay=0.95,
                              reg=reg, verbose=True)
                # training and validation set
                training_accuracy = stats['train_acc_history'][-1]
                print('training accuracy: %f' % (training_accuracy, ))
                validation_accuracy = stats['val_acc_history'][-1]
                print('validation accuracy: %f' % (validation_accuracy, ))
                results[(learning_rate, reg, hidden_size, batch_size)] = (training_accuracy, validation_accuracy)
                if validation_accuracy > best_val:
                    best_val = validation_accuracy
                    best_net = net

# Print out results.
for lr, reg, hidden_size, batch_size in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, hidden_size, batch_size)]
    print('lr %e reg %e hidden_size %e batch_size %e train accuracy: %f val accuracy: %f' % (
                lr, reg, hidden_size, batch_size, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross-validation: %f' % best_val)

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.savefig('nn_vis_grid_test.png')
    plt.show()

show_net_weights(best_net)

test_acc =  (np.sum(best_net.predict(X_test) * y_test, axis = 1)).mean()
print('Test accuracy: ', test_acc)
