import os
import shutil
import pandas as pd

# Load the Excel file
excel_file = "/Users/srivatsavkannan/Datasets/CombinedSpecies.xlsx"  # Change this to your actual file name
df = pd.read_excel(excel_file, usecols=[0, 2, 3], header=None)  # Read only necessary columns
df.columns = ["ID", "Name", "Scientific"]  # Ensure column names are correct

# Directory where the bird files are stored
source_directory = "/Users/srivatsavkannan/Datasets/Srivatsav"  # Change this to the actual path of your bird files
destination_directory = "/Users/srivatsavkannan/Datasets/Dataset"  # Base directory for sorted files

# Ensure the destination directory exists
os.makedirs(destination_directory, exist_ok=True)

# Process each row in the DataFrame
for _, row in df.iterrows():
    bird_id = str(row["ID"])  # Convert to string to match filenames
    folder_name = str(row["Name"])+' - '+str(row["Scientific"])


    # Create the folder for this name
    folder_path = os.path.join(destination_directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Move matching files
    for file in os.listdir(source_directory):
        if bird_id in file:  # Assuming filenames contain the ID
            src_path = os.path.join(source_directory, file)
            dest_path = os.path.join(folder_path, file)
            shutil.copy(src_path, dest_path)

print("Files organized successfully.")
