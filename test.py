import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = tf.keras.models.load_model('bird1.h5')

# Paths
test_folder = '/Users/srivatsavkannan/Datasets/BirdSong/classification_test'
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

# Prepare test data and labels
test_data = []
test_labels = []

for label in species:
    folder_path = os.path.join(test_folder, label)
    for file in os.listdir(folder_path):
        if file.endswith('.wav'):
            file_path = os.path.join(folder_path, file)
            spectrogram = wav_to_spectrogram(file_path)
            spectrogram_resized = tf.image.resize(spectrogram[..., np.newaxis], img_size)
            spectrogram_rgb = tf.image.grayscale_to_rgb(spectrogram_resized)  # Convert grayscale to RGB
            test_data.append(spectrogram_rgb.numpy())
            test_labels.append(label)

# Convert to numpy arrays
test_data = np.array(test_data)
test_labels = np.array(test_labels)

# Encode labels
label_encoder = LabelEncoder()
test_labels_encoded = label_encoder.fit_transform(test_labels)

test_labels_categorical = to_categorical(test_labels_encoded, num_classes=len(species))

# Predict
predictions = model.predict(test_data)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels_categorical, axis=1)

# Generate classification report
report = classification_report(true_classes, predicted_classes, target_names=species, digits=4)
print("Classification Report:\n", report)

# Generate confusion matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=species, yticklabels=species)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
