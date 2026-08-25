import os
import pandas as pd

def count_wav_files(folder_path):
    all_bird_data = {}
    total_count = 0

    for bird_name in os.listdir(folder_path):
        if not bird_name.startswith(tuple(v for v in ['ForestOwlet', 'BanasuraLaughingthrush', 'BugunLiocichla', 'Jerdon'])):
            continue

        subdirs = [
            'macauley',
            'Xeno Canto/A',
            'Xeno Canto/B',
            'Xeno Canto/C',
            'Xeno Canto/D',
            'Xeno Canto/E'
        ]

        bird_subdir_counts = {}
        bird_total = 0

        for subdir in subdirs:
            full_path = os.path.join(folder_path, bird_name, subdir)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                count = len(os.listdir(full_path))
            else:
                count = 0
            bird_subdir_counts[subdir] = count
            bird_total += count

        bird_subdir_counts['total'] = bird_total
        all_bird_data[bird_name] = bird_subdir_counts
        total_count += bird_total

    return all_bird_data, total_count


# Main execution
main_folder = '/Users/srivatsavkannan/Datasets/Bird Sound/Dataset/'
bird_data, grand_total = count_wav_files(main_folder)

# Sort by total descending
sorted_bird_data = sorted(bird_data.items(), key=lambda x: x[1]['total'], reverse=True)

# Print per bird
for bird_name, counts in sorted_bird_data:
    print(f"\n{bird_name}: Total = {counts['total']}")
    for subdir, count in counts.items():
        if subdir != 'total':
            print(f"  {subdir}: {count}")

# Print overall total
print(f"\nOverall Total: {grand_total}")
