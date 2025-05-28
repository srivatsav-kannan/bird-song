import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path



def convert(INPUT_DIR, OUTPUT_DIR):
    # Parameters
    SAMPLE_RATE = 22050
    IMG_SIZE = (256, 256)

    def save_spectrogram(file_path, output_path):
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Create figure
            fig = plt.figure(figsize=(IMG_SIZE[0] / 100, IMG_SIZE[1] / 100), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            librosa.display.specshow(S_dB, sr=sr, fmax=8000, ax=ax, cmap='magma')

            # Save to PNG
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

    # Walk through all files
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.m4a')):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, INPUT_DIR)
                output_path = os.path.join(OUTPUT_DIR, os.path.splitext(rel_path)[0] + '.png')
                save_spectrogram(input_path, output_path)

    print("Spectrogram conversion complete.")

# Input and output directories
INPUT = '/Users/srivatsavkannan/Datasets/BirdSong/classification_train'
OUTPUT = '/Users/srivatsavkannan/Datasets/BirdSong/classification_train_spectrogram'
convert(INPUT, OUTPUT)

INPUT = '/Users/srivatsavkannan/Datasets/BirdSong/classification_test'
OUTPUT = '/Users/srivatsavkannan/Datasets/BirdSong/classification_test_spectrogram'
convert(INPUT, OUTPUT)
