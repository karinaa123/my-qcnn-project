
import data
import Training
import QCNN_circuit
import Hierarchical_circuit
import numpy as np
import os
import matplotlib.pyplot as plt  # Needed for plotting


def accuracy_test(predictions, labels, cost_fn, binary=True):
    if cost_fn == 'mse':
        if binary == True:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 1:
                    acc = acc + 1
            return acc / len(labels)

        else:
            acc = 0
            for l, p in zip(labels, predictions):
                if np.abs(l - p) < 0.5:
                    acc = acc + 1
            return acc / len(labels)

    elif cost_fn == 'cross_entropy':
        acc = 0
        for l, p in zip(labels, predictions):
            if p[0] > p[1]:
                P = 0
            else:
                P = 1
            if P == l:
                acc = acc + 1
        return acc / len(labels)


def Encoding_to_Embedding(Encoding):
    # Amplitude Embedding / Angle Embedding
    if Encoding == 'resize256':
        Embedding = 'Amplitude'
    elif Encoding == 'pca8':
        Embedding = 'Angle'
    elif Encoding == 'autoencoder8':
        Embedding = 'Angle'

    # Amplitude Hybrid Embedding
    # 4 qubit block
    elif Encoding == 'pca32-1':
        Embedding = 'Amplitude-Hybrid4-1'
    elif Encoding == 'autoencoder32-1':
        Embedding = 'Amplitude-Hybrid4-1'

    elif Encoding == 'pca32-2':
        Embedding = 'Amplitude-Hybrid4-2'
    elif Encoding == 'autoencoder32-2':
        Embedding = 'Amplitude-Hybrid4-2'

    elif Encoding == 'pca32-3':
        Embedding = 'Amplitude-Hybrid4-3'
    elif Encoding == 'autoencoder32-3':
        Embedding = 'Amplitude-Hybrid4-3'

    elif Encoding == 'pca32-4':
        Embedding = 'Amplitude-Hybrid4-4'
    elif Encoding == 'autoencoder32-4':
        Embedding = 'Amplitude-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca16-1':
        Embedding = 'Amplitude-Hybrid2-1'
    elif Encoding == 'autoencoder16-1':
        Embedding = 'Amplitude-Hybrid2-1'

    elif Encoding == 'pca16-2':
        Embedding = 'Amplitude-Hybrid2-2'
    elif Encoding == 'autoencoder16-2':
        Embedding = 'Amplitude-Hybrid2-2'

    elif Encoding == 'pca16-3':
        Embedding = 'Amplitude-Hybrid2-3'
    elif Encoding == 'autoencoder16-3':
        Embedding = 'Amplitude-Hybrid2-3'

    elif Encoding == 'pca16-4':
        Embedding = 'Amplitude-Hybrid2-4'
    elif Encoding == 'autoencoder16-4':
        Embedding = 'Amplitude-Hybrid2-4'

    # Angular HybridEmbedding
    # 4 qubit block
    elif Encoding == 'pca30-1':
        Embedding = 'Angular-Hybrid4-1'
    elif Encoding == 'autoencoder30-1':
        Embedding = 'Angular-Hybrid4-1'

    elif Encoding == 'pca30-2':
        Embedding = 'Angular-Hybrid4-2'
    elif Encoding == 'autoencoder30-2':
        Embedding = 'Angular-Hybrid4-2'

    elif Encoding == 'pca30-3':
        Embedding = 'Angular-Hybrid4-3'
    elif Encoding == 'autoencoder30-3':
        Embedding = 'Angular-Hybrid4-3'

    elif Encoding == 'pca30-4':
        Embedding = 'Angular-Hybrid4-4'
    elif Encoding == 'autoencoder30-4':
        Embedding = 'Angular-Hybrid4-4'

    # 2 qubit block
    elif Encoding == 'pca12-1':
        Embedding = 'Angular-Hybrid2-1'
    elif Encoding == 'autoencoder12-1':
        Embedding = 'Angular-Hybrid2-1'

    elif Encoding == 'pca12-2':
        Embedding = 'Angular-Hybrid2-2'
    elif Encoding == 'autoencoder12-2':
        Embedding = 'Angular-Hybrid2-2'

    elif Encoding == 'pca12-3':
        Embedding = 'Angular-Hybrid2-3'
    elif Encoding == 'autoencoder12-3':
        Embedding = 'Angular-Hybrid2-3'

    elif Encoding == 'pca12-4':
        Embedding = 'Angular-Hybrid2-4'
    elif Encoding == 'autoencoder12-4':
        Embedding = 'Angular-Hybrid2-4'

    # Two Gates Compact Encoding
    elif Encoding == 'pca16-compact':
        Embedding = 'Angle-compact'
    elif Encoding == 'autoencoder16-compact':
        Embedding = 'Angle-compact'
    return Embedding


def Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit, cost_fn, binary=True):
    I = len(Unitaries)
    J = len(Encodings)

    if not os.path.exists('Result'):
        os.makedirs('Result')

    for i in range(I):
        for j in range(J):
            f = open('Result/result.txt', 'a')

            U = Unitaries[i]
            U_params = U_num_params[i]
            Encoding = Encodings[j]
            Embedding = Encoding_to_Embedding(Encoding)

            print("\n--------------------------------------------------------------------------------")
            print(f"Processing: {dataset} | {U} | {Encoding} ({Embedding})")
            print("--------------------------------------------------------------------------------")

            X_train, X_test, Y_train, Y_test = data.data_load_and_process(dataset, classes=classes,
                                                                          feature_reduction=Encoding, binary=binary)

            print("\nStarting Quantum Circuit Training...")
            # Updated to receive validation history and steps
            loss_history_train, loss_history_val, val_steps, trained_params = Training.circuit_training(
                X_train, Y_train, X_test, Y_test, U, U_params, Embedding, circuit, cost_fn
            )

            print("\nEvaluating on Test Set...")
            if circuit == 'QCNN':
                predictions = [QCNN_circuit.QCNN(x, trained_params, U, U_params, Embedding, cost_fn) for x in X_test]
            elif circuit == 'Hierarchical':
                predictions = [
                    Hierarchical_circuit.Hierarchical_classifier(x, trained_params, U, U_params, Embedding, cost_fn) for
                    x in X_test]

            accuracy = accuracy_test(predictions, Y_test, cost_fn, binary)
            print(f"Final Accuracy for {U} {Encoding} : {accuracy:.4f}")

            # ----------------------------------------------------------------
            # PLOTTING THE GRAPH
            # ----------------------------------------------------------------
            plt.figure(figsize=(10, 6))
            plt.plot(loss_history_train, label='Training Loss', alpha=0.5)
            # Plot validation loss at the specific steps it was calculated
            plt.plot(val_steps, loss_history_val, label='Validation Loss', linewidth=2, color='red')
            plt.title(f"Loss History: {U} - {Encoding}")
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.legend()
            plt.grid(True)

            # Save the plot
            plot_filename = f"Result/LossPlot_{U}_{Encoding}.png"
            plt.savefig(plot_filename)
            plt.close()
            print(f"Graph saved to {plot_filename}")

            # Save Results to text file
            f.write(f"Configuration: {circuit}, {U}, {Encoding}, {cost_fn}\n")
            f.write(f"Final Accuracy: {accuracy}\n")
            f.write(f"Training Loss History: {loss_history_train}\n")
            f.write(f"Validation Loss History: {loss_history_val}\n")
            f.write("\n")
            f.close()

    print("\nBenchmarking Complete! Check the 'Result' folder for graphs and text report.")