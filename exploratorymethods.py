import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json

def plot_class_distribution(train_df, val_df, class_mapping):
    """
    Plots the class distribution for training and validation datasets.

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
    
    sns.countplot(data=train_df, x="class_name", order=class_names.values(), ax=axes[0], palette="pastel")
    axes[0].set_title("Training Dataset Class Distribution")
    axes[0].set_xlabel("Class Name")
    axes[0].set_ylabel("Frequency")
    axes[0].tick_params(axis='x', rotation=45)

    sns.countplot(data=val_df, x="class_name", order=class_names.values(), ax=axes[1], palette="muted")
    axes[1].set_title("Validation Dataset Class Distribution")
    axes[1].set_xlabel("Class Name")
    axes[1].tick_params(axis='x', rotation=45)

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
    Analyze the number of images filmed during the day, night, and other times.
    
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
    plt.figure(figsize=(8, 5))
    time_of_day_counts.plot(kind="bar", color=["skyblue", "orange", "green", "purple"])
    plt.title("Distribution of Images by Time of Day")
    plt.xlabel("Time of Day")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


