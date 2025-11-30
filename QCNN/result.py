# This generates the results of the benchmarking code
import Benchmarking

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]
Encodings: ['resize256', 'pca8', 'autoencoder8', 'pca16-compact', 'autoencoder16-compact', 'pca32-1', 'autoencoder32-1',
            'pca16-1', 'autoencoder16-1', 'pca30-1', 'autoencoder30-1', 'pca12-1', 'autoencoder12-1']
dataset: 'mnist' or 'fashion_mnist' or 'miniddsm'
circuit: 'QCNN' or 'Hierarchical'
cost_fn: 'mse' or 'cross_entropy'
Note: when using 'mse' as cost_fn binary="True" is recommended, when using 'cross_entropy' as cost_fn must be binary="False".
"""

# -----------------------------------------------------------------------------------------
# CONFIGURATION FOR MINI-DDSM BREAST CANCER DETECTION
# -----------------------------------------------------------------------------------------

dataset = 'miniddsm'

# Classes (Normal/Benign are merged into 'Benign' by data.py)
classes = ['Cancer', 'Benign']

# --- RUNNING MULTIPLE CIRCUITS ---
# The code will train these one by one and save all results to Result/result.txt
Unitaries = ['U_SU4', 'U_SO4']
U_num_params = [15, 6]

# Convolutional Autoencoder (32 features)
Encodings = ['autoencoder32-1']

cost_fn = 'cross_entropy'
binary = True

# -----------------------------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------------------------
print(f"Running Experiment: {dataset} | {classes} | {Encodings}")
print(f"Circuits to benchmark: {Unitaries}")

Benchmarking.Benchmarking(dataset, classes, Unitaries, U_num_params, Encodings, circuit='QCNN', cost_fn=cost_fn, binary=binary)