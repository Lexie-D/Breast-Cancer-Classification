import os
import pandas as pd
from sklearn.model_selection import train_test_split
import glob

# Set dataset directory path
dataset_dir = "/content/Dataset_BUSI_with_GT"

# Gather image paths and labels
image_paths = []
labels = []

for label in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, label)
    if os.path.isdir(class_dir):
        for file in glob.glob(os.path.join(class_dir, "*.png")):
            image_paths.append(file)
            labels.append(label)

# Create DataFrame
df = pd.DataFrame({
    'Path': image_paths,
    'Label': labels
})

# Split 70% train, 15% val, 15% test
train_df, val_test_df = train_test_split(df, train_size=0.7, stratify=df['Label'], random_state=42)
val_df, test_df = train_test_split(val_test_df, train_size=0.5, stratify=val_test_df['Label'], random_state=42)

# Save splits
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Confirm splits
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")