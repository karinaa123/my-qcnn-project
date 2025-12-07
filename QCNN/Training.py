import os
import autograd.numpy as np
from autograd import grad
from Nesterov_optimizer import NesterovOptimizer


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


def cross_entropy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        c = 1e-15
        loss = loss + (-l * np.log(p + c) - (1 - l) * np.log(1 - p + c))
    loss = loss / len(labels)
    return loss


def accuracy_test(predictions, labels, binary=True):
    if binary:
        acc = 0
        for l, p in zip(labels, predictions):
            if np.abs(l - p) < 0.5:
                acc = acc + 1
        return acc / len(labels)
    else:
        acc = 0
        for l, p in zip(labels, predictions):
            if np.argmax(l) == np.argmax(p):
                acc = acc + 1
        return acc / len(labels)


def circuit_training(X_train, Y_train, X_test, Y_test, U, U_params, embedding_type, circuit, cost_fn, binary):
    if binary:
        steps = 200
        batch_size = 25
    else:
        steps = 10000  # Matches your log output
        batch_size = 32

    # Optimizer initialization
    opt = NesterovOptimizer(step_size=0.01)
    params = U_params

    # History tracking
    loss_history_train = []
    loss_history_val = []
    val_steps = []

    print(f"Starting Training for {steps} iterations...")

    for i in range(steps):
        # Generate training batch indices
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[j] for j in batch_index]
        Y_batch = [Y_train[j] for j in batch_index]

        # Optimization step
        # Note: Depending on your exact QCNN setup, the cost function arguments might vary slightly.
        # This matches the standard implementation structure.
        def cost_wrapper(params):
            predictions = [circuit(x, params, U, embedding_type, cost_fn) for x in X_batch]
            return cost_fn(Y_batch, predictions)

        params = opt.step(cost_wrapper, params)

        # Validation Step (Every 10 or 20 iterations)
        if i % 20 == 0:
            val_index = np.random.randint(0, len(X_test), (batch_size,))
            X_val_batch = [X_test[j] for j in val_index]
            Y_val_batch = [Y_test[j] for j in val_index]
            # Calculate Training Loss for current batch
            train_predictions = [circuit(x, params, U, embedding_type, cost_fn) for x in X_batch]
            train_loss = cost_fn(Y_batch, train_predictions)
            loss_history_train.append(train_loss)

            # Calculate Validation Loss
            val_predictions = [circuit(x, params, U, embedding_type, cost_fn) for x in X_val_batch]
            val_loss = cost_fn(Y_val_batch, val_predictions)
            loss_history_val.append(val_loss)

            val_steps.append(i)

            print(f"Iteration: {i} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return loss_history_train, loss_history_val, val_steps, params