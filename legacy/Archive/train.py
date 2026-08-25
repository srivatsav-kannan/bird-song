import os
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shutil

# Paths
classification_folder = '/Users/srivatsavkannan/Datasets/BirdSong/classification'
species = ['bewickii', 'cardinalis', 'melodia', 'migratorius', 'polyglottos']

# Parameters
img_size = (128, 128)
sampling_rate = 22050

# Function to convert wav to spectrogram image
def wav_to_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=sampling_rate)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db

# Create train and test folders
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(test_folder, exist_ok=True)
# for label in species:
#     os.makedirs(os.path.join(train_folder, label), exist_ok=True)
#     os.makedirs(os.path.join(test_folder, label), exist_ok=True)
#
# # Prepare filenames for splitting
# filenames = []
# for label in species:
#     folder_path = os.path.join(classification_folder, label)
#     for file in os.listdir(folder_path):
#         if file.endswith('.wav'):
#             filenames.append((file, label))

# # Shuffle and split filenames into train and test sets
# train_files, test_files = train_test_split(filenames, test_size=0.2, random_state=42)
#
# # Distribute files into train and test folders
# for file, label in train_files:
#     src = os.path.join(classification_folder, label, file)
#     dst = os.path.join(train_folder, label, file)
#     shutil.copy(src, dst)
#
# for file, label in test_files:
#     src = os.path.join(classification_folder, label, file)
#     dst = os.path.join(test_folder, label, file)
#     shutil.copy(src, dst)

# Prepare data and labels
data = []
labels = []

for label in species:
    folder_path = os.path.join(classification_folder, label)
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            spectrogram = wav_to_spectrogram(file_path)
            spectrogram_resized = tf.image.resize(spectrogram[..., np.newaxis], img_size)
            spectrogram_3ch = tf.image.grayscale_to_rgb(spectrogram_resized)  # Convert to 3 channels

            data.append(spectrogram_3ch.numpy())
            labels.append(label)

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data, labels_categorical, test_size=0.2, random_state=42
)

# Load ResNet50 model with pretrained weights
# base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
# for layer in base_model.layers:
#     layer.trainable = False
#
# # Build the model
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dense(128, activation='relu')(x)
# x = Dropout(0.3)(x)
# predictions = Dense(len(species), activation='softmax')(x)
# model = Model(inputs=base_model.input, outputs=predictions)
#
# # Compile model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='max',
#                                                restore_best_weights=True)
# # Train model
# history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test), callbacks=[early_stop])
# model.save('bird1.h5')
model = tf.keras.models.load_model('bird1.h5')
# Evaluate model
print(X_test.shape)
plt.imshow(X_test[0])
plt.show()
print(y_test.shape)
eval_result = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {eval_result[1] * 100:.2f}%")

# # Plot training history
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.legend()
# plt.show()
