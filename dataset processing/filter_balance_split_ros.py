import os
import shutil
import random
import numpy as np
import math
from pathlib import Path

# === Paths ===
SOURCE_DIR = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated'
BALANCED_DIR = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset_Curated_Split_Balanced'

# === Settings ===
VALID_EXT = ('.wav', '.mp3', '.m4a')
# VALID_EXT = ('.wav')
OVERSAMPLE_PRIORITY = ['macauley', 'Xeno Canto/A', 'Xeno Canto/B', 'Xeno Canto/C', 'Xeno Canto/D']
TRAIN_SPLIT = 0.8
SEED = 42
random.seed(SEED)

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
    return len(OVERSAMPLE_PRIORITY)  # lowest priority if not found

def undersample_files(files, target_count):
    """Keep the highest priority files first, trim low priority"""
    sorted_files = sorted(files, key=get_priority_index)
    return sorted_files[:target_count]

def oversample_files(files, target_count):
    """Oversample using source preference"""
    if len(files) >= target_count:
        return undersample_files(files, target_count)

    diff = target_count - len(files)

    preferred = []
    for priority in OVERSAMPLE_PRIORITY:
        matches = [f for f in files if f'/{priority}/' in f]
        preferred.extend(matches)

    if not preferred:
        preferred = files

    extra = [random.choice(preferred) for _ in range(diff)]
    return files + extra

# === Main Function ===
def balance_and_split_dataset():
    bird_files = {}

    # Step 1: Collect all species and audio files
    for bird in os.listdir(SOURCE_DIR):
        bird_path = os.path.join(SOURCE_DIR, bird)
        if not os.path.isdir(bird_path):
            continue

        files = get_all_audio_files(bird_path)
        if len(files) >= 2:  # Need at least 2 to split into train/val
            bird_files[bird] = files

    # Step 2: Train/test split without overlap
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
            print(f"Excluded {bird}: insufficient split (train={len(train_split)}, val={len(val_split)})")

    # Step 3: Get mean count for balancing
    train_count = int(np.max([len(v) for v in train_map.values()]))
    val_count = int(np.max([len(v) for v in test_map.values()]))
    print(f"Target files per species — Train: {train_count}, Val: {val_count}")

    # Step 4: Balance and write files
    for split_name, data_map, count in [('train', train_map, train_count), ('val', test_map, val_count)]:
        for bird, files in data_map.items():
            split_dir = os.path.join(BALANCED_DIR, split_name, bird)
            os.makedirs(split_dir, exist_ok=True)

            balanced_files = oversample_files(files, count)

            for idx, f in enumerate(balanced_files):
                ext = Path(f).suffix
                dst_path = os.path.join(split_dir, f"{idx}{ext}")
                shutil.copy(f, dst_path)

# === Run ===
if __name__ == "__main__":
    print("Starting dataset balancing and splitting...")
    balance_and_split_dataset()
    print("Finished: Balanced and split dataset is ready.")
