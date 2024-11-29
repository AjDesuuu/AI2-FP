import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import cv2
from collections import Counter

def plot_class_distribution(train_df, val_df, class_mapping):
    """
    Plots the class distribution for training and validation datasets with specific numbers displayed on bars.

    Parameters:
        train_df (DataFrame): DataFrame containing training YOLO labels.
        val_df (DataFrame): DataFrame containing validation YOLO labels.
        class_mapping (dict): Dictionary mapping class IDs to class names.

    Returns:
        None
    """
    # Create readable class names
    class_names = {v: k for k, v in class_mapping.items()}
    train_df["class_name"] = train_df["class_id"].map(class_names)
    val_df["class_name"] = val_df["class_id"].map(class_names)

    # Plot class distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    
    train_ax = sns.countplot(data=train_df, x="class_name", order=class_names.values(), ax=axes[0], palette="pastel")
    train_ax.set_title("Training Dataset Class Distribution")
    train_ax.set_xlabel("Class Name")
    train_ax.set_ylabel("Frequency")
    train_ax.tick_params(axis='x', rotation=45)
    train_ax.bar_label(train_ax.containers[0])  # Add labels to the bars

    val_ax = sns.countplot(data=val_df, x="class_name", order=class_names.values(), ax=axes[1], palette="muted")
    val_ax.set_title("Validation Dataset Class Distribution")
    val_ax.set_xlabel("Class Name")
    val_ax.tick_params(axis='x', rotation=45)
    val_ax.bar_label(val_ax.containers[0])  # Add labels to the bars

    plt.tight_layout()
    plt.show()


def plot_bbox_size_distribution(train_df, val_df):
    """
    Plots the distribution of bounding box sizes (width and height) for both training and validation datasets.

    Parameters:
        train_df (DataFrame): DataFrame containing training YOLO labels.
        val_df (DataFrame): DataFrame containing validation YOLO labels.

    Returns:
        None
    """
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    sns.histplot(train_df["bbox_width"], kde=True, bins=30, ax=axes[0], label="Width", color="blue", alpha=0.7)
    sns.histplot(train_df["bbox_height"], kde=True, bins=30, ax=axes[0], label="Height", color="orange", alpha=0.7)
    axes[0].set_title("Training Dataset Bounding Box Sizes")
    axes[0].set_xlabel("Bounding Box Size (pixels)")
    axes[0].legend()

    sns.histplot(val_df["bbox_width"], kde=True, bins=30, ax=axes[1], label="Width", color="blue", alpha=0.7)
    sns.histplot(val_df["bbox_height"], kde=True, bins=30, ax=axes[1], label="Height", color="orange", alpha=0.7)
    axes[1].set_title("Validation Dataset Bounding Box Sizes")
    axes[1].set_xlabel("Bounding Box Size (pixels)")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_aspect_ratio_distribution(train_df, val_df):
    """
    Plots the distribution of aspect ratios for bounding boxes in training and validation datasets.

    Parameters:
        train_df (DataFrame): DataFrame containing training YOLO labels.
        val_df (DataFrame): DataFrame containing validation YOLO labels.

    Returns:
        None
    """
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    sns.histplot(train_df["aspect_ratio"], kde=True, bins=30, ax=axes[0], color="green", alpha=0.7)
    axes[0].set_title("Training Dataset Aspect Ratio Distribution")
    axes[0].set_xlabel("Aspect Ratio (Width / Height)")
    
    sns.histplot(val_df["aspect_ratio"], kde=True, bins=30, ax=axes[1], color="green", alpha=0.7)
    axes[1].set_title("Validation Dataset Aspect Ratio Distribution")
    axes[1].set_xlabel("Aspect Ratio (Width / Height)")

    plt.tight_layout()
    plt.show()

def plot_bbox_area_distribution(train_df, val_df):
    """
    Plots the distribution of bounding box areas for both training and validation datasets.

    Parameters:
        train_df (DataFrame): DataFrame containing training YOLO labels.
        val_df (DataFrame): DataFrame containing validation YOLO labels.

    Returns:
        None
    """
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    sns.histplot(train_df["area"], kde=True, bins=30, ax=axes[0], color="purple", alpha=0.7)
    axes[0].set_title("Training Dataset Bounding Box Area Distribution")
    axes[0].set_xlabel("Bounding Box Area (pixels)")

    sns.histplot(val_df["area"], kde=True, bins=30, ax=axes[1], color="purple", alpha=0.7)
    axes[1].set_title("Validation Dataset Bounding Box Area Distribution")
    axes[1].set_xlabel("Bounding Box Area (pixels)")

    plt.tight_layout()
    plt.show()


def analyze_time_of_day(annotations_path):
    """
    Analyze the number of images filmed during the day, night, and other times, with specific numbers on bars.
    
    Parameters:
    - annotations_path (str): Path to the annotations JSON file.
    
    Returns:
    - None (Displays a bar plot of the time-of-day distribution)
    """
    # Load annotations
    with open(annotations_path, 'r') as f:
        annotations = json.load(f)
    
    # Extract 'time of day' from attributes
    time_of_day_list = [
        annotation.get("attributes", {}).get("timeofday", "Unknown")
        for annotation in annotations
    ]
    
    # Count occurrences
    time_of_day_counts = pd.Series(time_of_day_list).value_counts()
    
    # Plot the distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = time_of_day_counts.plot(kind="bar", color=["skyblue", "orange", "green", "purple"], ax=ax)
    ax.set_title("Distribution of Images by Time of Day")
    ax.set_xlabel("Time of Day")
    ax.set_ylabel("Number of Images")
    ax.bar_label(bars.containers[0])  # Add labels to the bars
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)
    plt.show()



def check_file_format_distribution(images_train_path, images_val_path, images_test_path):
    """
    Checks and visualizes the number of images per file format across train, val, and test directories.
    Parameters:
        images_train_path (str): Path to the training images directory.
        images_val_path (str): Path to the validation images directory.
        images_test_path (str): Path to the test images directory.
    Returns:
        None (Displays a bar plot of file format distribution)
    """
    # Combine all file paths for analysis
    all_paths = {
        "Train": images_train_path,
        "Validation": images_val_path,
        "Test": images_test_path,
    }
    
    # Initialize a Counter for file formats
    format_counts = Counter()
    # Analyze each dataset split
    for split_name, path in all_paths.items():
        formats = [
            os.path.splitext(file)[1].lower()
            for root, _, files in os.walk(path)
            for file in files
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))
        ]
        for fmt, count in Counter(formats).items():
            format_counts[f"{split_name} ({fmt})"] = count
    # Convert to DataFrame for plotting
    format_df = pd.DataFrame(format_counts.items(), columns=["Split and Format", "Count"])
    format_df = format_df.sort_values(by="Count", ascending=False)
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Split and Format", y="Count", data=format_df, palette="viridis")
    plt.title("Number of Images per File Format by Dataset Split")
    plt.xlabel("Dataset Split and File Format")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print the counts
    print("File Format Distribution Across Splits:")
    print(format_df.to_string(index=False))

def plot_image_dimensions_histogram(images_train_path, images_val_path, images_test_path):
    """
    Plots histograms for image heights and widths across train, val, and test datasets.

    Parameters:
        images_train_path (str): Path to the training images directory.
        images_val_path (str): Path to the validation images directory.
        images_test_path (str): Path to the test images directory.

    Returns:
        None (Displays histograms for image heights and widths)
    """
    # Combine all file paths for analysis
    all_paths = {
        "Train": images_train_path,
        "Validation": images_val_path,
        "Test": images_test_path,
    }
    
    # Initialize dictionaries for heights and widths
    heights = {"Train": [], "Validation": [], "Test": []}
    widths = {"Train": [], "Validation": [], "Test": []}

    # Collect image dimensions
    for split_name, path in all_paths.items():
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        heights[split_name].append(h)
                        widths[split_name].append(w)

    # Plot histograms
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)

    for i, split_name in enumerate(["Train", "Validation", "Test"]):
        sns.histplot(heights[split_name], kde=True, bins=30, ax=axes[0, i], color="blue", alpha=0.7)
        axes[0, i].set_title(f"{split_name} Image Heights")
        axes[0, i].set_xlabel("Height (pixels)")
        axes[0, i].set_ylabel("Frequency")

        sns.histplot(widths[split_name], kde=True, bins=30, ax=axes[1, i], color="green", alpha=0.7)
        axes[1, i].set_title(f"{split_name} Image Widths")
        axes[1, i].set_xlabel("Width (pixels)")
        axes[1, i].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()
