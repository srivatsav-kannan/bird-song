import os
import pandas as pd
import shutil

# Paths
wav_folder = '/Users/srivatsavkannan/Datasets/BirdSong/wavfiles'
metadata_file = '/Users/srivatsavkannan/Datasets/BirdSong/bird_songs_metadata.csv'
classification_folder = '/Users/srivatsavkannan/Datasets/BirdSong/classification'

# Load metadata
metadata = pd.read_csv(metadata_file)

# Ensure classification folder exists
os.makedirs(classification_folder, exist_ok=True)

# Process each row in the metadata
idd = 0
prev = 'f'
for index, row in metadata.iterrows():
    file_id = str(row['filename'])
    species_name = row['species']


    # Source and destination paths
    source_file = os.path.join(wav_folder, f"{file_id}")
    species_folder = os.path.join(classification_folder, species_name)
    destination_file = os.path.join(species_folder, f"{file_id}")


    # Create species folder if it doesn't exist
    os.makedirs(species_folder, exist_ok=True)

    # Move the file to the species folder
    if os.path.exists(source_file):
        shutil.move(source_file, destination_file)
        print(f"File {file_id} moved to {species_folder}")
    else:
        print(f"File {file_id} not found in {wav_folder}")

print("Files have been organized by species.")