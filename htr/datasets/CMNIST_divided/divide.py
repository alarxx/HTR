import os
import shutil
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict

def split_dataset(input_dir, output_dir, test_size=0.2):
    """
    Splits dataset into train and test preserving class distribution.

    Parameters:
    - input_dir: str, path to the CMNIST folder containing class subfolders.
    - output_dir: str, path to the output folder.
    - test_size: float, proportion of the data for testing (default 0.2).
    """
    # Paths for train and test
    train_dir = os.path.join(output_dir, 'train')
    test_dir = os.path.join(output_dir, 'test')

    # Ensure output directories exist
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    class_files = defaultdict(list)
    
    # Collect all files for each class
    for class_name in os.listdir(input_dir):
        class_path = os.path.join(input_dir, class_name)
        if os.path.isdir(class_path):
            files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            class_files[class_name].extend(files)

    # Split data for each class and copy
    for class_name, files in class_files.items():
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)

        # Create class subfolders in train and test directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Copy files to train directory
        for file in train_files:
            shutil.copy(file, os.path.join(train_dir, class_name))

        # Copy files to test directory
        for file in test_files:
            shutil.copy(file, os.path.join(test_dir, class_name))

        print(f"Class '{class_name}': {len(train_files)} train, {len(test_files)} test")

def display_class_distribution(directory, title="Class Distribution"):
    """
    Displays the distribution of classes in the given directory as a bar chart.

    Parameters:
    - directory: str, path to the directory containing class subfolders.
    - title: str, title for the plot.
    """
    class_counts = defaultdict(int)
    
    # Count files in each class directory
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            file_count = len([f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))])
            class_counts[class_name] = file_count

    # Plot class distribution
    classes = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts)
    plt.xlabel("Classes")
    plt.ylabel("Number of Files")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_dir = './CMNIST'
    output_dir = './split'
    test_size = 0.2  # 20% for testing

    split_dataset(input_dir, output_dir, test_size)
    print("Dataset splitting completed!")

    # Display distributions for train and test sets
    print("\nTrain set distribution:")
    display_class_distribution(os.path.join(input_dir), title="Original Class Distribution")

    # Display distributions for train and test sets
    print("\nTrain set distribution:")
    display_class_distribution(os.path.join(output_dir, 'train'), title="Train Class Distribution")
    
    print("\nTest set distribution:")
    display_class_distribution(os.path.join(output_dir, 'test'), title="Test Class Distribution")
