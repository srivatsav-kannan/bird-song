import os
import shutil
import random
import numpy as np
import math
import librosa
import soundfile as sf
from pathlib import Path

# === Paths ===
SOURCE_DIR = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated'
BALANCED_DIR = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated_Split_Balanced_Aug'

# === Settings ===
VALID_EXT = ('.wav', '.mp3', '.m4a')
OVERSAMPLE_PRIORITY = ['macauley', 'Xeno Canto/A', 'Xeno Canto/B', 'Xeno Canto/C', 'Xeno Canto/D']
TRAIN_SPLIT = 0.8
SEED = 42
random.seed(SEED)
AUGMENT_SR = 22050  # Standard sampling rate for augmentation

# === Helper Functions ===
def get_all_audio_files(root):
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(VALID_EXT):
                full_path = os.path.join(dirpath, f)
                all_files.append(full_path)
    return all_files

def get_priority_index(filepath):
    for i, priority in enumerate(OVERSAMPLE_PRIORITY):
        if f'/{priority}/' in filepath:
            return i
    return len(OVERSAMPLE_PRIORITY)

# def undersample_files(files, target_count):
#     sorted_files = sorted(files, key=get_priority_index)
#     return sorted_files[:target_count]

def add_gaussian_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

def add_pink_noise(audio, noise_level=0.005):
    # Generate pink noise using Voss-McCartney algorithm
    b = [0.02109238, 0.07113478, 0.68873558]
    a = [1, -1.73472577, 0.7660066]
    pink = np.random.randn(len(audio))
    pink = np.convolve(pink, b, mode='same') / np.convolve(np.ones(len(audio)), a, mode='same')
    pink = pink / np.max(np.abs(pink)) * noise_level
    return audio + pink

def augment_audio(original_path, noise_type='gaussian'):
    y, sr = librosa.load(original_path, sr=AUGMENT_SR)
    if noise_type == 'gaussian':
        y_aug = add_gaussian_noise(y)
    elif noise_type == 'pink':
        y_aug = add_pink_noise(y)
    else:
        raise ValueError("Unknown noise type")
    return y_aug, sr

def oversample_files_with_augmentation(files, target_count, bird, split_dir):
    retained = list(files)
    needed = target_count - len(retained)
    augmented_paths = []

    for i in range(needed):
        original = random.choice(files)
        noise_type = 'gaussian' if i % 2 == 0 else 'pink'
        y_aug, sr = augment_audio(original, noise_type=noise_type)

        ext = Path(original).suffix
        aug_filename = f"aug_{i}{ext}"
        aug_path = os.path.join(split_dir, aug_filename)

        sf.write(aug_path, y_aug, sr)
        augmented_paths.append(aug_path)

    return retained + augmented_paths

# === Main Function ===
def balance_and_split_dataset():
    bird_files = {}

    # Step 1: Collect all species and audio files
    for bird in os.listdir(SOURCE_DIR):
        bird_path = os.path.join(SOURCE_DIR, bird)
        if not os.path.isdir(bird_path):
            continue

        files = get_all_audio_files(bird_path)
        if len(files) >= 2:
            bird_files[bird] = files

    # Step 2: Train/test split
    train_map = {}
    test_map = {}

    for bird, files in bird_files.items():
        random.shuffle(files)
        split_idx = math.ceil(len(files) * TRAIN_SPLIT)
        train_split = files[:split_idx]
        val_split = files[split_idx:]

        if len(train_split) > 0 and len(val_split) > 0:
            train_map[bird] = train_split
            test_map[bird] = val_split
        else:
            print(f"Excluded {bird}: insufficient split")

    # Step 3: Compute max class count
    train_target = int(np.max([len(v) for v in train_map.values()]))
    val_target = int(np.max([len(v) for v in test_map.values()]))
    print(f"Target per species — Train: {train_target}, Val: {val_target}")

    # Step 4: Balance and copy/augment
    for split_name, data_map, target in [('train', train_map, train_target), ('val', test_map, val_target)]:
        for bird, files in data_map.items():
            split_dir = os.path.join(BALANCED_DIR, split_name, bird)
            os.makedirs(split_dir, exist_ok=True)

            balanced_files = oversample_files_with_augmentation(files, target, bird, split_dir)

            for idx, f in enumerate(balanced_files):
                dst_ext = Path(f).suffix
                dst_path = os.path.join(split_dir, f"{idx}{dst_ext}")
                if not os.path.exists(dst_path):
                    shutil.copy(f, dst_path)

# === Run ===
if __name__ == "__main__":
    print("Starting dataset balancing and splitting with augmentation...")
    balance_and_split_dataset()
    print("Finished: Augmented and balanced dataset ready.")
