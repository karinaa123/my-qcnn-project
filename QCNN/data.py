# Loads and Processes the data that will be used in QCNN and Hierarchical Classifier Training
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import cv2
import glob
import os
from sklearn.model_selection import train_test_split

# ============================================================
# DATASET CONFIGURATION - Mini-DDSM2
# ============================================================
# We use expanduser("~") to automatically find your home directory (e.g., /home/yourname/)
MINIDDSM_BASE_DIR = os.path.expanduser("~/datasets/miniddsm2/MINI-DDSM-Complete-PNG-16")
# ============================================================

pca32 = ['pca32-1', 'pca32-2', 'pca32-3', 'pca32-4']
autoencoder32 = ['autoencoder32-1', 'autoencoder32-2', 'autoencoder32-3', 'autoencoder32-4']
pca30 = ['pca30-1', 'pca30-2', 'pca30-3', 'pca30-4']
autoencoder30 = ['autoencoder30-1', 'autoencoder30-2', 'autoencoder30-3', 'autoencoder30-4']
pca16 = ['pca16-1', 'pca16-2', 'pca16-3', 'pca16-4', 'pca16-compact']
autoencoder16 = ['autoencoder16-1', 'autoencoder16-2', 'autoencoder16-3', 'autoencoder16-4', 'autoencoder16-compact']
pca12 = ['pca12-1', 'pca12-2', 'pca12-3', 'pca12-4']
autoencoder12 = ['autoencoder12-1', 'autoencoder12-2', 'autoencoder12-3', 'autoencoder12-4']


def data_load_and_process(dataset, classes=[0, 1], feature_reduction='resize256', binary=True):
    # ----------------------------------------------------------------------------------------------
    # 1. LOAD DATASET (Updated for Mini-DDSM)
    # ----------------------------------------------------------------------------------------------
    if dataset == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

    elif dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0


    elif dataset == 'miniddsm':
        data_path = MINIDDSM_BASE_DIR
        images = []
        labels = []
        # ------------------------------------------------------------
        # 1. COMBINE CLASSES
        # We map 'Cancer' to 1.
        # We map BOTH 'Benign' and 'Normal' to 0.
        # This effectively merges them into a single "Benign" class.
        # ------------------------------------------------------------
        class_map = {
            'Cancer': 1,
            'Benign': 0,
            'Normal': 0
        }
        print(f"Scanning for images in {data_path}...")
        # Load images from folders
        for folder_name, label_val in class_map.items():
            folder_full_path = os.path.join(data_path, folder_name)
            if not os.path.exists(folder_full_path):
                print(f"Warning: Folder not found: {folder_full_path}")
                continue
            # Recursive search for .png files
            search_pattern = os.path.join(folder_full_path, "**", "*.png")
            file_list = glob.glob(search_pattern, recursive=True)
            print(f"Found {len(file_list)} images in {folder_name} (mapped to Label: {label_val})")
            for file_path in file_list:
                try:
                    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # img = cv2.resize(img, (28, 28))
                        img = cv2.resize(img, (128, 128))
                        images.append(img)
                        labels.append(label_val)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        x_data = np.array(images)
        y_data = np.array(labels)
        if len(x_data) == 0:
            raise ValueError(f"No images found! Check path: {MINIDDSM_BASE_DIR}")
        # ------------------------------------------------------------
        # 2. FORCE 50/50 BALANCE (Undersampling)
        # ------------------------------------------------------------

        # Find indices for Cancer (Label 1) and Combined Benign (Label 0)
        idx_cancer = np.where(y_data == 1)[0]
        idx_benign = np.where(y_data == 0)[0]
        print(f"\n--- Balancing Data ---")
        print(f"Total Cancer images: {len(idx_cancer)}")
        print(f"Total Benign (Normal + Benign) images: {len(idx_benign)}")

        # Determine the target size (the size of the smaller class)
        target_size = min(len(idx_cancer), len(idx_benign))
        print(f"Target size per class: {target_size} (50/50 Split)")

        # Randomly sample 'target_size' indices from both groups
        np.random.seed(42)  # Fixed seed for reproducibility
        idx_cancer_balanced = np.random.choice(idx_cancer, target_size, replace=False)
        idx_benign_balanced = np.random.choice(idx_benign, target_size, replace=False)
        # Combine and shuffle the selected indices
        balanced_indices = np.concatenate([idx_cancer_balanced, idx_benign_balanced])
        np.random.shuffle(balanced_indices)
        # Apply selection to the actual data
        x_data = x_data[balanced_indices]
        y_data = y_data[balanced_indices]
        print(f"Final Dataset Size: {len(x_data)} images ({np.sum(y_data == 1)} Cancer, {np.sum(y_data == 0)} Benign)")
        # ------------------------------------------------------------

        # Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)
        x_data = np.expand_dims(x_data, axis=-1)
        # Normalize to 0-1 range (Standard for neural networks)
        x_data = x_data / 255.0
        # Split into Train and Test
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
        print(f"Train/Test Split: {len(x_train)} training, {len(x_test)} testing samples.")
    # ----------------------------------------------------------------------------------------------
    # 2. FILTER CLASSES (Skipped for Mini-DDSM)
    # ----------------------------------------------------------------------------------------------
    if dataset != 'miniddsm':
        if classes == 'odd_even':
            odd = [1, 3, 5, 7, 9]
            X_train = x_train
            X_test = x_test
            if binary == False:
                Y_train = [1 if y in odd else 0 for y in y_train]
                Y_test = [1 if y in odd else 0 for y in y_test]
            elif binary == True:
                Y_train = [1 if y in odd else -1 for y in y_train]
                Y_test = [1 if y in odd else -1 for y in y_test]

        elif classes == '>4':
            greater = [5, 6, 7, 8, 9]
            X_train = x_train
            X_test = x_test
            if binary == False:
                Y_train = [1 if y in greater else 0 for y in y_train]
                Y_test = [1 if y in greater else 0 for y in y_test]
            elif binary == True:
                Y_train = [1 if y in greater else -1 for y in y_train]
                Y_test = [1 if y in greater else -1 for y in y_test]

        else:
            x_train_filter_01 = np.where((y_train == classes[0]) | (y_train == classes[1]))
            x_test_filter_01 = np.where((y_test == classes[0]) | (y_test == classes[1]))

            X_train, X_test = x_train[x_train_filter_01], x_test[x_test_filter_01]
            Y_train, Y_test = y_train[x_train_filter_01], y_test[x_test_filter_01]

            if binary == False:
                Y_train = [1 if y == classes[0] else 0 for y in Y_train]
                Y_test = [1 if y == classes[0] else 0 for y in Y_test]
            elif binary == True:
                Y_train = [1 if y == classes[0] else -1 for y in Y_train]
                Y_test = [1 if y == classes[0] else -1 for y in Y_test]
    else:
        # For Mini-DDSM, the variables are already set correctly by our custom loader
        X_train, X_test, Y_train, Y_test = x_train, x_test, y_train, y_test

    # ----------------------------------------------------------------------------------------------
    # 3. FEATURE REDUCTION (Convolutional Autoencoder)
    # ----------------------------------------------------------------------------------------------

    if feature_reduction == 'resize256':
        X_train = tf.image.resize(X_train[:], (256, 1)).numpy()
        X_test = tf.image.resize(X_test[:], (256, 1)).numpy()
        X_train, X_test = tf.squeeze(X_train).numpy(), tf.squeeze(X_test).numpy()
        return X_train, X_test, Y_train, Y_test

    elif feature_reduction == 'pca8' or feature_reduction in pca32 \
            or feature_reduction in pca30 or feature_reduction in pca16 or feature_reduction in pca12:

        X_train = tf.image.resize(X_train[:], (784, 1)).numpy()
        X_test = tf.image.resize(X_test[:], (784, 1)).numpy()
        X_train, X_test = tf.squeeze(X_train), tf.squeeze(X_test)

        if feature_reduction == 'pca8':
            pca = PCA(8)
        elif feature_reduction in pca32:
            pca = PCA(32)
        elif feature_reduction in pca30:
            pca = PCA(30)
        elif feature_reduction in pca16:
            pca = PCA(16)
        elif feature_reduction in pca12:
            pca = PCA(12)

        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        if feature_reduction == 'pca8' or feature_reduction == 'pca16-compact' or \
                feature_reduction in pca30 or feature_reduction in pca12:
            X_train, X_test = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min())), \
                              (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))
        return X_train, X_test, Y_train, Y_test

    elif feature_reduction == 'autoencoder8' or feature_reduction in autoencoder32 \
            or feature_reduction in autoencoder30 or feature_reduction in autoencoder16 or feature_reduction in autoencoder12:
        if feature_reduction == 'autoencoder8':
            latent_dim = 8
        elif feature_reduction in autoencoder32:
            latent_dim = 32
        elif feature_reduction in autoencoder30:
            latent_dim = 30
        elif feature_reduction in autoencoder16:
            latent_dim = 16
        elif feature_reduction in autoencoder12:
            latent_dim = 12

        # --- CONVOLUTIONAL AUTOENCODER ---
        # class Autoencoder(Model):
        #    def __init__(self, latent_dim):
        #        super(Autoencoder, self).__init__()
        #        self.latent_dim = latent_dim

        #        self.encoder = tf.keras.Sequential([
        #           layers.InputLayer(input_shape=(28, 28, 1)),
        #            layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
        #            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
        #            layers.Flatten(),
        #            layers.Dense(latent_dim, activation='relu'),
        #        ])
            # --- 针对 128x128 输入优化的 Autoencoder ---
        class Autoencoder(Model):
            def __init__(self, latent_dim):
                super(Autoencoder, self).__init__()
                self.latent_dim = latent_dim
                # ENCODER: 把 128x128 的图片压缩成 32 个特征
                self.encoder = tf.keras.Sequential([
                    layers.InputLayer(input_shape=(128, 128, 1)),
                    # 128x128 -> 64x64
                    layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2),
                    # 64x64 -> 32x32
                    layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
                    # 32x32 -> 16x16
                    layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),
                    # 16x16 -> 8x8
                    # 2048 -> 512
                    layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=2),
                    layers.Flatten(),
                    # 8*8*4 = 256
                    layers.Dense(latent_dim, activation='relu'),
                ])

                    # DECODER: 把 32 个特征还原回 128x128 (用于训练 Autoencoder)
                self.decoder = tf.keras.Sequential([
                    layers.Dense(8 * 8 * 4, activation='relu'),
                    layers.Reshape((8, 8, 4)),
                    # 8x8 -> 16x16
                    layers.Conv2DTranspose(4, (3, 3), activation='relu', padding='same', strides=2),
                    # 16x16 -> 32x32
                    layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same', strides=2),
                    # 32x32 -> 64x64
                    layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same', strides=2),
                    # 64x64 -> 128x128
                    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same', strides=2),
                ])
            def call(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return decoded

        autoencoder = Autoencoder(latent_dim)
        autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

        print("Training Autoencoder on Image Data...")
        autoencoder.fit(X_train, X_train,
                        epochs=50,
                        batch_size=32,
                        shuffle=True,
                        validation_data=(X_test, X_test))

        X_train = autoencoder.encoder(X_train).numpy()
        X_test = autoencoder.encoder(X_test).numpy()
        # Check reconstruction quality
        reconstructed = autoencoder.predict(X_test[:5])
        print(f"Reconstruction MSE: {np.mean((X_test[:5] - reconstructed) ** 2)}")

        # Visualize
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(X_test[i].squeeze(), cmap='gray')
            axes[0, i].set_title('Original')
            axes[1, i].imshow(reconstructed[i].squeeze(), cmap='gray')
            axes[1, i].set_title('Reconstructed')
        plt.savefig('autoencoder_check.png')

        if feature_reduction == 'autoencoder8' or feature_reduction == 'autoencoder16-compact' or \
                feature_reduction in autoencoder32 or \
                feature_reduction in autoencoder30 or feature_reduction in autoencoder12:
            X_train, X_test = (X_train - X_train.min()) * (np.pi / (X_train.max() - X_train.min())), \
                              (X_test - X_test.min()) * (np.pi / (X_test.max() - X_test.min()))

        return X_train, X_test, Y_train, Y_test