import os
import shutil
import random

# Define source and target directories
source_dir = './RiceFolder/Rice_Image_Dataset/Karacadag'
train_dir = './Rice/train/Karacadag'
val_dir = './Rice/val/Karacadag'
test_dir = './Rice/test/Karacadag'

# Get list of all files
all_files = os.listdir(source_dir)
random.shuffle(all_files)

# Calculate splitting points
train_split = int(0.6 * len(all_files))
val_split = int(0.8 * len(all_files))

# Split files
train_files = all_files[:train_split]
val_files = all_files[train_split:val_split]
test_files = all_files[val_split:]

# Function to move files
def move_files(files, target_dir):
    for file in files:
        shutil.move(os.path.join(source_dir, file), os.path.join(target_dir, file))

# Move files to respective directories
move_files(train_files, train_dir)
print("train")
move_files(val_files, val_dir)
print("val")
move_files(test_files, test_dir)

print("test")