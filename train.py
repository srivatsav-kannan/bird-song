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

train_dir = "/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated_Balanced_Split_Converted/train"
val_dir = "/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated_Balanced_Split_Converted/val"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16
EPOCHS = 20
AUTOTUNE = tf.data.experimental.AUTOTUNE
class_names = sorted(os.listdir(train_dir))[1:]
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

training = True

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


def testing(dataset, model):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        predictions = model.predict(images)  # Get predictions as a 1D array
        binary_predictions = np.argmax(predictions, axis=-1)
        y_true.extend(labels.numpy())
        y_pred.extend(binary_predictions)

    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate accuracy, precision, recall, and F1-score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Generate classification report with 4 significant figures
    report = classification_report(y_true, y_pred, digits=4)

    # Output metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Classification Report:")
    print(report)

    # Create and display confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


model = tf.keras.models.load_model('bird_effnetb7.keras')
testing(val_ds, model)
