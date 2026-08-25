import os
import shutil
import random
import numpy as np
import math
import librosa
import soundfile as sf
import torch
from pathlib import Path

# === Paths ===
SOURCE_DIR = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated'
BALANCED_DIR = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated_Balanced_Split_MixUp'

# === Settings ===
VALID_EXT = ('.wav', '.mp3', '.m4a')
OVERSAMPLE_PRIORITY = ['macauley', 'Xeno Canto/A', 'Xeno Canto/B', 'Xeno Canto/C', 'Xeno Canto/D']
TRAIN_SPLIT = 0.8
SEED = 42
random.seed(SEED)
AUGMENT_SR = 22050

# === Helper Functions ===
def get_all_audio_files(root):
    all_files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(VALID_EXT):
                all_files.append(os.path.join(dirpath, f))
    return all_files

def get_priority_index(filepath):
    for i, priority in enumerate(OVERSAMPLE_PRIORITY):
        if f'/{priority}/' in filepath:
            return i
    return len(OVERSAMPLE_PRIORITY)

def load_audio_tensor(file_path, sr=AUGMENT_SR):
    y, _ = librosa.load(file_path, sr=sr)
    return torch.tensor(y, dtype=torch.float32)

def mixup_audio(file1, file2):
    x1 = load_audio_tensor(file1)
    x2 = load_audio_tensor(file2)

    # Pad to same length
    max_len = max(x1.shape[0], x2.shape[0])
    x1 = torch.nn.functional.pad(x1, (0, max_len - x1.shape[0]))
    x2 = torch.nn.functional.pad(x2, (0, max_len - x2.shape[0]))

    t = random.uniform(0, 1)
    return (t * x1 + (1 - t) * x2).numpy()

def oversample_files_with_mixup(files, target_count, bird, split_dir):
    retained = list(files)
    needed = target_count - len(retained)
    augmented_paths = []

    for i in range(needed):
        f1, f2 = random.sample(files, 2)
        mixed = mixup_audio(f1, f2)

        ext = Path(f1).suffix
        orig_name = Path(f1).stem
        aug_filename = f"{orig_name}_1{ext}"
        aug_path = os.path.join(split_dir, aug_filename)

        sf.write(aug_path, mixed, AUGMENT_SR)
        augmented_paths.append(aug_path)

    return retained + augmented_paths

# === Main Function ===
def balance_and_split_dataset():
    bird_files = {}

    # Step 1: Collect species and files
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

    # Step 3: Determine max counts
    train_target = int(np.max([len(v) for v in train_map.values()]))
    val_target = int(np.max([len(v) for v in test_map.values()]))
    print(f"Target per species — Train: {train_target}, Val: {val_target}")

    # Step 4: Balance + copy/augment
    for split_name, data_map, target in [('train', train_map, train_target), ('val', test_map, val_target)]:
        for bird, files in data_map.items():
            split_dir = os.path.join(BALANCED_DIR, split_name, bird)
            os.makedirs(split_dir, exist_ok=True)

            if split_name == 'train':
                balanced_files = oversample_files_with_mixup(files, target, bird, split_dir)
            else:
                balanced_files = files

            for idx, f in enumerate(balanced_files):
                ext = Path(f).suffix
                dst_path = os.path.join(split_dir, f"{idx}{ext}")
                if not os.path.exists(dst_path):
                    shutil.copy(f, dst_path)

# === Run ===
if __name__ == "__main__":
    print("Starting dataset balancing and splitting with MixUp augmentation...")
    balance_and_split_dataset()
    print("Finished: MixUp-augmented balanced dataset ready.")
