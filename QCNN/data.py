# Loads and Processes the data that will be used in QCNN Training
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import os
from PIL import Image
import pandas as pd

# ============================================================
# DATASET CONFIGURATION - Mini-DDSM2
# ============================================================
# Update this path to where you downloaded Mini-DDSM2
MINIDDSM_BASE_DIR = os.path.expanduser("~/datasets/miniddsm2/MINI-DDSM-Complete-PNG-16")
MINIDDSM_CSV = os.path.join(MINIDDSM_BASE_DIR, "mini-DDSM-Complete-PNG-16.csv")
# ============================================================


def load_miniddsm2(base_dir, csv_path, img_size=(28, 28), max_samples_per_class=None):
    """
    Load Mini-DDSM2 dataset for binary breast cancer classification.
    """
    print(f"Loading Mini-DDSM2 from: {base_dir}")

    # Check if dataset exists
    if not os.path.exists(base_dir):
        raise FileNotFoundError(
            f"Mini-DDSM2 dataset not found at {base_dir}\n"
            f"Please download it first using:\n"
            f"  kaggle datasets download -d cheddad/miniddsm2"
        )

    # Fallback to folder scan immediately if CSV issues arise or for simplicity
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV metadata file not found. Scanning directories instead...")
        return load_miniddsm2_from_folders(base_dir, img_size, max_samples_per_class)

    # Read CSV metadata
    try:
        df = pd.read_csv(csv_path)
        print(f"CSV loaded: {len(df)} entries")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return load_miniddsm2_from_folders(base_dir, img_size, max_samples_per_class)

    # ... (CSV logic omitted for brevity, deferring to robust folder scan below) ...
    # If you prefer CSV loading, ensure the mask check is added there too.
    # For now, we route to folder scan to guarantee mask exclusion.
    return load_miniddsm2_from_folders(base_dir, img_size, max_samples_per_class)


def load_miniddsm2_from_folders(base_dir, img_size=(28, 28), max_samples_per_class=None):
    print("Scanning folders for images...")
    images = []
    labels = []

    benign_folders = ['Benign', 'Normal']
    malignant_folders = ['Cancer']

    for folder_name in benign_folders:
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.exists(folder_path):
            count = load_images_from_folder(folder_path, images, labels, label=0,
                                            img_size=img_size, max_count=max_samples_per_class)
            print(f"  Loaded {count} Benign images from {folder_name}")

    for folder_name in malignant_folders:
        folder_path = os.path.join(base_dir, folder_name)
        if os.path.exists(folder_path):
            count = load_images_from_folder(folder_path, images, labels, label=1,
                                            img_size=img_size, max_count=max_samples_per_class)
            print(f"  Loaded {count} Malignant images from {folder_name}")

    if len(images) == 0:
        raise RuntimeError(f"No images found in {base_dir}.")

    return process_and_split_data(images, labels)


def load_images_from_folder(folder_path, images, labels, label, img_size, max_count=None):
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if max_count and count >= max_count:
                return count

            # --- KEY FIX HERE: EXCLUDE MASKS ---
            # We check if 'mask' is in the filename.
            if filename.lower().endswith('.png') and 'mask' not in filename.lower():
                try:
                    img_path = os.path.join(root, filename)
                    # Convert to Grayscale ('L')
                    img = Image.open(img_path).convert('L')
                    img = img.resize(img_size)
                    img_array = np.array(img)

                    images.append(img_array)
                    labels.append(label)
                    count += 1
                except Exception:
                    continue
    return count


def process_and_split_data(images, labels):
    """
    Process loaded images and split into train/test sets (8:2).
    """
    images = np.array(images)
    labels = np.array(labels)

    print(f"\nTotal loaded: {len(images)} images")
    print(f"Benign: {np.sum(labels == 0)}, Malignant: {np.sum(labels == 1)}")

    # Shuffle
    indices = np.random.permutation(len(images))
    images = images[indices]
    labels = labels[indices]

    # ---------------------------------------------------------
    # SPLIT LOGIC: 80% Train, 20% Test
    # ---------------------------------------------------------
    split_idx = int(0.8 * len(images))

    x_train = images[:split_idx]
    y_train = labels[:split_idx]

    x_test = images[split_idx:]
    y_test = labels[split_idx:]

    # Normalize and add channel dimension
    x_train = x_train[..., np.newaxis] / 255.0
    x_test = x_test[..., np.newaxis] / 255.0

    print(f"\n✅ Split Results (8:2):")
    print(f"   Train: {len(x_train)}")
    print(f"   Test:  {len(x_test)}")

    return x_train, y_train, x_test, y_test


def data_load_and_process(dataset, classes=[0, 1], feature_reduction='resize256', binary=True):
    """
    Load and preprocess data.
    Returns: X_train, X_test, Y_train, Y_test
    """

    # 1. LOAD DATASET
    if dataset in ['fashion_mnist', 'mnist']:
        if dataset == 'fashion_mnist':
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train[..., np.newaxis] / 255.0
        x_test = x_test[..., np.newaxis] / 255.0

    elif dataset == 'miniddsm2':
        # Load all data (max_samples_per_class=None to disable limit)
        x_train, y_train, x_test, y_test = load_miniddsm2(
            MINIDDSM_BASE_DIR, MINIDDSM_CSV, img_size=(28, 28), max_samples_per_class=None
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 2. FILTER CLASSES (For MNIST/Fashion)
    if dataset in ['mnist', 'fashion_mnist']:
        if classes == 'odd_even':
            odd = [1, 3, 5, 7, 9]
            X_train, X_test = x_train, x_test
        elif classes == '>4':
            greater = [5, 6, 7, 8, 9]
            X_train, X_test = x_train, x_test
        else:
            x_train_filter = np.where((y_train == classes[0]) | (y_train == classes[1]))
            x_test_filter = np.where((y_test == classes[0]) | (y_test == classes[1]))
            X_train, X_test = x_train[x_train_filter], x_test[x_test_filter]
            y_train, y_test = y_train[x_train_filter], y_test[x_test_filter]
    else:
        X_train, X_test = x_train, x_test

    # 3. LABEL CONVERSION
    if binary:
        if dataset == 'miniddsm2':
            # 0=Benign, 1=Malignant.
            # Map 0 -> 1 (Target), 1 -> -1 (Non-target)
            Y_train = [1 if y == 0 else -1 for y in y_train]
            Y_test = [1 if y == 0 else -1 for y in y_test]
        else:
            if classes == 'odd_even':
                odd = [1, 3, 5, 7, 9]
                Y_train = [1 if y in odd else -1 for y in y_train]
                Y_test = [1 if y in odd else -1 for y in y_test]
            elif classes == '>4':
                greater = [5, 6, 7, 8, 9]
                Y_train = [1 if y in greater else -1 for y in y_train]
                Y_test = [1 if y in greater else -1 for y in y_test]
            else:
                Y_train = [1 if y == classes[0] else -1 for y in y_train]
                Y_test = [1 if y == classes[0] else -1 for y in y_test]
    else:
        Y_train = y_train
        Y_test = y_test

    # 4. FEATURE REDUCTION
    print(f"Applying feature reduction: {feature_reduction}")

    if feature_reduction == 'resize256':
        def resize_fn(X):
            # Resizes to 256 pixels total (e.g. 16x16 flattened or just linear resize)
            # The original code resized to (256, 1).
            X_r = tf.image.resize(X[:], (256, 1)).numpy()
            return tf.squeeze(X_r).numpy()

        X_train = resize_fn(X_train)
        X_test = resize_fn(X_test)

    elif 'pca' in feature_reduction: # pca8, pca16
        X_train = tf.squeeze(tf.image.resize(X_train[:], (784, 1))).numpy()
        X_test = tf.squeeze(tf.image.resize(X_test[:], (784, 1))).numpy()

        n_components = 16 if '16' in feature_reduction else 8
        pca = PCA(n_components)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Scale to [0, pi] using TRAINING statistics
        min_val, max_val = X_train.min(), X_train.max()
        scale = np.pi / (max_val - min_val)

        X_train = (X_train - min_val) * scale
        X_test = (X_test - min_val) * scale

    elif 'autoencoder' in feature_reduction: # autoencoder8, autoencoder16
        latent_dim = 16 if '16' in feature_reduction else 8

        class Autoencoder(Model):
            def __init__(self, latent_dim):
                super(Autoencoder, self).__init__()
                self.encoder = tf.keras.Sequential([
                    layers.Flatten(),
                    layers.Dense(latent_dim, activation='relu'),
                ])
                self.decoder = tf.keras.Sequential([
                    layers.Dense(784, activation='sigmoid'),
                    layers.Reshape((28, 28))
                ])
            def call(self, x):
                return self.decoder(self.encoder(x))

        autoencoder = Autoencoder(latent_dim)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        autoencoder.fit(X_train, X_train, epochs=10, shuffle=True,
                        validation_data=(X_test, X_test), verbose=1)

        X_train = autoencoder.encoder(X_train).numpy()
        X_test = autoencoder.encoder(X_test).numpy()

        # Scale to [0, pi] using TRAINING statistics
        min_val, max_val = X_train.min(), X_train.max()
        scale = np.pi / (max_val - min_val)

        X_train = (X_train - min_val) * scale
        X_test = (X_test - min_val) * scale

    return X_train, X_test, Y_train, Y_test