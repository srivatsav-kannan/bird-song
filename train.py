import json
import os
import sys

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Define constants
print(tf.config.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_dir = "/Users/srivatsavkannan/Datasets/BirdSong/Dataset_Curated_Balanced_Converted/train"
val_dir = "/Users/srivatsavkannan/Datasets/BirdSong/Dataset_Curated_Balanced_Converted/val"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 20
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = sorted(list(s for s in os.listdir(train_dir) if s != '.DS_Store'))
print(class_names)
print(len(class_names))

train_ds = image_dataset_from_directory(
    train_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

val_ds = image_dataset_from_directory(
    val_dir,
    seed=123,
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE)

# Configure dataset for performance
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

training = False

if training:
    # Define the input shape for the model (standard image dimensions with 3 color channels)
    input_shape = (256, 256, 3)

    # Load the pretrained ResNet50 model without the classification head (include_top=False)
    # and with ImageNet weights; use this as the feature extractor
    base_model = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape
    )

    # # Freeze the base model to prevent its weights from being updated during training
    base_model.trainable = False

    # Define the model's input layer with the specified shape
    inputs = tf.keras.Input(shape=input_shape)

    # Pass the inputs through the pretrained base model without training it
    x = base_model(inputs, training=False)

    # Add a dense layer to learn additional representations from extracted features
    x = tf.keras.layers.Dense(512, activation='relu')(x)

    # Apply global average pooling to reduce spatial dimensions to a single vector
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Final output layer with sigmoid activation for binary classification
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

    # Create the full model by specifying inputs and outputs
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.summary()

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=100, mode='max',
                                                  restore_best_weights=True)

    history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=[early_stop])
    model.save('bird_effnetb7.keras')

    with open("bird_effnetb7.json", "w") as f:
        json.dump(history.history, f)


import textwrap

# Updated class names with spaces
class_names = [
    "Banasura Laughingthrush",
    "Bugun Liocichla",
    "Forest Owlet",
    "Jerdon's Courser"
]

def format_labels(labels, width=12):
    """Wraps long labels into multiple lines without splitting words."""
    return ['\n'.join(textwrap.wrap(label, width, break_long_words=False, break_on_hyphens=False))
            for label in labels]


def testing(dataset, model, title):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)
        binary_predictions = np.argmax(predictions, axis=-1)
        y_true.extend(labels.numpy())
        y_pred.extend(binary_predictions)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    report = classification_report(y_true, y_pred, digits=4)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)

    # Format labels for better display
    wrapped_labels = format_labels(class_names)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wrapped_labels, yticklabels=wrapped_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def testing_soft_voting(dataset, model1, model2):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        preds1 = model1.predict(images) * 0.4
        preds2 = model2.predict(images) * 0.6
        avg_preds = preds1 + preds2
        final_preds = np.argmax(avg_preds, axis=-1)

        y_true.extend(labels.numpy())
        y_pred.extend(final_preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    report = classification_report(y_true, y_pred, digits=4)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    wrapped_labels = format_labels(class_names)

    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wrapped_labels, yticklabels=wrapped_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Soft Voting Ensemble Confusion Matrix')
    plt.tight_layout()
    plt.show()


model_effnet = tf.keras.models.load_model('bird_effnetb7.keras')
model_resnet = tf.keras.models.load_model('bird_resnet.keras')

testing(val_ds, model_resnet, title='ResNet50 Confusion Matrix')
testing(val_ds, model_effnet, title='EffNetB7 Confusion Matrix')
testing_soft_voting(val_ds, model_effnet, model_resnet)
#


with open("bird_effnetb7.json", "r") as f:
    history = json.load(f)

# Plot training & validation accuracy
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()