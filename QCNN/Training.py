# Implementation of Quantum circuit training procedure
# cross entrophy requires labels to be 0 and -1,
# rn labels are -1 and 1
# accept the Test data (X_test, Y_test) and calculate validation loss every 50 iterations.
# Implementation of Quantum circuit training procedure
import QCNN_circuit
import Hierarchical_circuit
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c_entropy = l * (anp.log(p[l])) + (1 - l) * anp.log(1 - p[1 - l])
        loss = loss + c_entropy
    return -1 * loss


def cost(params, X, Y, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        predictions = [QCNN_circuit.QCNN(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x in X]
    elif circuit == 'Hierarchical':
        predictions = [
            Hierarchical_circuit.Hierarchical_classifier(x, params, U, U_params, embedding_type, cost_fn=cost_fn) for x
            in X]

    if cost_fn == 'mse':
        loss = square_loss(Y, predictions)
    elif cost_fn == 'cross_entropy':
        loss = cross_entropy(Y, predictions)
    return loss


# ----------------------------------------------------------------------------------
# TRAINING PARAMETERS
# ----------------------------------------------------------------------------------
steps = 2000
learning_rate = 0.001
batch_size = 16


# Updated to accept Validation Data (X_test, Y_test)
def circuit_training(X_train, Y_train, X_test, Y_test, U, U_params, embedding_type, circuit, cost_fn):
    if circuit == 'QCNN':
        if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
            total_params = U_params * 3
        else:
            total_params = U_params * 3 + 2 * 3
    elif circuit == 'Hierarchical':
        total_params = U_params * 7

    params = np.random.randn(total_params, requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=learning_rate)

    # Storage for history
    loss_history_train = []
    loss_history_val = []
    steps_history = []

    print(f"Starting Training for {steps} iterations...")

    for it in range(steps):
        # 1. Train on a batch
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        params, cost_train = opt.step_and_cost(
            lambda v: cost(v, X_batch, Y_batch, U, U_params, embedding_type, circuit, cost_fn), params)

        loss_history_train.append(cost_train)

        # 2. Validation Check (Every 50 steps)
        # We check on a random batch of the test set to save time (calculating full test set is too slow)
        if it % 50 == 0:
            val_batch_index = np.random.randint(0, len(X_test), (batch_size,))
            X_val_batch = [X_test[i] for i in val_batch_index]
            Y_val_batch = [Y_test[i] for i in val_batch_index]

            cost_val = cost(params, X_val_batch, Y_val_batch, U, U_params, embedding_type, circuit, cost_fn)
            loss_history_val.append(cost_val)
            steps_history.append(it)

            print(f"Iteration: {it:4d} | Train Cost: {cost_train:.4f} | Val Cost: {cost_val:.4f}")

    return loss_history_train, loss_history_val, steps_history, params