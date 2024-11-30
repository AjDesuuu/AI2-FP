import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np
from collections import Counter
from pathlib import Path
import math
import random
from PIL import Image



def plot_class_distribution(train_df, val_df, class_mapping):
    """
    Plots the class distribution for training and validation datasets with specific numbers displayed on bars
    and prints numerical class distributions.

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
    
    train_ax = sns.countplot(
        data=train_df,
        x="class_name",
        order=class_names.values(),
        ax=axes[0],
        hue="class_name",  # Assign hue to the same variable as x
        palette="pastel",
        dodge=False,  # Avoid duplicate bars
        legend=False  # Suppress legend
    )
    train_ax.set_title("Training Dataset Class Distribution")
    train_ax.set_xlabel("Class Name")
    train_ax.set_ylabel("Frequency")
    train_ax.tick_params(axis='x', rotation=45)
    train_ax.bar_label(train_ax.containers[0])  # Add labels to the bars

    val_ax = sns.countplot(
        data=val_df,
        x="class_name",
        order=class_names.values(),
        ax=axes[1],
        hue="class_name",  # Assign hue to the same variable as x
        palette="muted",
        dodge=False,  # Avoid duplicate bars
        legend=False  # Suppress legend
    )
    val_ax.set_title("Validation Dataset Class Distribution")
    val_ax.set_xlabel("Class Name")
    val_ax.tick_params(axis='x', rotation=45)
    val_ax.bar_label(val_ax.containers[0])  # Add labels to the bars

    plt.tight_layout()
    plt.show()

    # Numerical summary of class distributions
    train_counts = train_df["class_name"].value_counts()
    val_counts = val_df["class_name"].value_counts()

    print("\n### Class Distribution Summary ###")
    print("\nTraining Dataset:")
    print(train_counts.to_string())
    print("\nValidation Dataset:")
    print(val_counts.to_string())



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
    Checks and visualizes the number of images per file format across the train, val, and test datasets.
    Groups by file format and aggregates the counts across splits.

    Parameters:
        images_train_path (str): Path to the training images directory.
        images_val_path (str): Path to the validation images directory.
        images_test_path (str): Path to the testing images directory.

    Returns:
        None (Displays a bar plot of file format distribution)
    """
    # List of valid image file extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')

    def count_file_formats(directory):
        formats = []
        
        # Debugging: Check directory structure
        print(f"Scanning directory: {directory}")
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    formats.append(ext)
        
        if not formats:
            print(f"No valid image files found in {directory}.")
        
        return Counter(formats)

    # Count file formats for each dataset split
    train_counts = count_file_formats(images_train_path)
    val_counts = count_file_formats(images_val_path)
    test_counts = count_file_formats(images_test_path)

    # Combine all counts into one
    format_counts = train_counts + val_counts + test_counts

    # Handle case when no images are found
    if not format_counts:
        print("No image files were found across the specified directories.")
        return

    # Aggregate counts for the same formats (e.g., all .jpg files should be counted together)
    aggregated_counts = Counter()
    for format, count in format_counts.items():
        # Remove any leading dot (.) from the file extension to group by format
        clean_format = format.lstrip('.')
        aggregated_counts[clean_format] += count

    # Convert the counts into a DataFrame for plotting
    format_df = pd.DataFrame(aggregated_counts.items(), columns=["Format", "Count"])
    
    # Sort by count
    format_df = format_df.sort_values(by="Count", ascending=False)

    # Plot the distribution of file formats
    plt.figure(figsize=(12, 7))
    sns.barplot(data=format_df, x="Format", y="Count", palette="viridis")
    plt.title("Number of Images per File Format Across All Dataset Splits")
    plt.xlabel("File Format")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

    # Print the counts
    print("File Format Distribution Across All Dataset Splits:")
    print(format_df.to_string(index=False))


def detect_blurry_images(images_train_path, images_val_path, images_test_path, threshold=100.0):
    """
    Detects blurry images in train, val, and test datasets and visualizes the counts.

    Parameters:
        images_train_path (str): Path to the training images directory.
        images_val_path (str): Path to the validation images directory.
        images_test_path (str): Path to the test images directory.
        threshold (float): Threshold value for the Laplacian variance below which an image is considered blurry.

    Returns:
        None (Displays a bar plot of blurry and non-blurry images per dataset split)
    """
    # Combine all file paths for analysis
    all_paths = {
        "Train": images_train_path,
        "Validation": images_val_path,
        "Test": images_test_path,
    }
    
    # Initialize counters
    results = {"Dataset": [], "Blurry": [], "Non-Blurry": []}

    # Analyze each dataset split
    for split_name, path in all_paths.items():
        blurry_count = 0
        non_blurry_count = 0

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Compute Laplacian variance
                        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                        if laplacian_var < threshold:
                            blurry_count += 1
                        else:
                            non_blurry_count += 1

        # Append results
        results["Dataset"].append(split_name)
        results["Blurry"].append(blurry_count)
        results["Non-Blurry"].append(non_blurry_count)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot the results
    results_df.set_index("Dataset")[["Blurry", "Non-Blurry"]].plot(
        kind="bar", stacked=True, figsize=(10, 6), color=["red", "green"], alpha=0.8
    )
    plt.title("Blurry vs Non-Blurry Images by Dataset Split")
    plt.xlabel("Dataset Split")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=0)
    plt.legend(title="Image Quality")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print a summary
    print("Summary of Blurry Images:")
    print(results_df.to_string(index=False))


def display_image_grid(image_dir, num_images_to_show=9, num_cols=3, figsize_per_row=(15, 5), verbose=True, seed=None):
    """
    Enhanced image grid display with additional metadata and analysis.
    
    Parameters:
    -----------
    image_dir : str
        Path to the directory containing images
    num_images_to_show : int, optional
        Number of images to display (default is 9)
    num_cols : int, optional
        Number of columns in the grid (default is 3)
    figsize_per_row : tuple, optional
        Figure size multiplier per row (default is (15, 5))
    verbose : bool, optional
        If True, print additional information about the images
    seed : int or None, optional
        A seed for the random number generator to ensure reproducibility (default is None)
    
    Returns:
    --------
    None
        Displays the image grid using matplotlib
    """
    # List all images in the directory
    image_files = os.listdir(image_dir)
    
    # Verbose analysis
    if verbose:
        print("Dataset Image Inspection:")
        print(f"Total images in directory: {len(image_files)}")
    
    # Set the seed for reproducibility (if provided)
    if seed is not None:
        random.seed(seed)
        if verbose:
            print(f"Using seed: {seed}")
    
    # Select a random sample of images
    sample_images = random.sample(image_files, min(num_images_to_show, len(image_files)))
    
    if verbose:
        print(f"Images to be displayed: {len(sample_images)}")
    
    # Calculate grid dimensions
    num_rows = math.ceil(len(sample_images) / num_cols)
    
    # Create a figure to display the images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(figsize_per_row[0], figsize_per_row[1] * num_rows))
    
    # Flatten the axes array to make it easier to iterate
    axes = axes.flatten() if num_rows > 1 else axes
    
    # Loop through each image to display it
    for i, img_file in enumerate(sample_images):
        img_path = os.path.join(image_dir, img_file)
        img = Image.open(img_path)
        
        # Display the image
        axes[i].imshow(img)
        
        # Add more detailed title
        if verbose:
            img_size = img.size
            img_mode = img.mode
            axes[i].set_title(f"{img_file}\n{img_size[0]}x{img_size[1]} {img_mode}", fontsize=10)
        else:
            axes[i].set_title(f"Image: {img_file}")
        
        axes[i].axis('off')  # Turn off axis
    
    # Hide any unused subplots
    for j in range(len(sample_images), len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


def plot_image_resolution_distribution(image_dir):
    """
    Provides visualization of image resolution distribution using histograms.
    
    Parameters:
    -----------
    image_dir : str
        Directory containing the dataset images.
    
    Returns:
    --------
    None
        Displays the plots for image resolution histograms.
    """
    image_widths = []
    image_heights = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            # Check for image files (common image extensions)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    
                    # Check if image is successfully read
                    if img is not None:
                        image_widths.append(img.shape[1])  # Width
                        image_heights.append(img.shape[0])  # Height
                except PermissionError as e:
                    print(f"Permission error accessing {img_path}: {e}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    
    # Create DataFrame for image resolutions
    image_resolution_df = pd.DataFrame({
        'Width': image_widths,
        'Height': image_heights
    })
    
    # Plot histograms for image dimensions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Width histogram
    sns.histplot(image_resolution_df['Width'], kde=True, bins=30, ax=axes[0], color="blue", alpha=0.7)
    axes[0].set_title("Image Width Distribution")
    axes[0].set_xlabel("Width (pixels)")
    axes[0].set_ylabel("Frequency")
    
    # Height histogram
    sns.histplot(image_resolution_df['Height'], kde=True, bins=30, ax=axes[1], color="green", alpha=0.7)
    axes[1].set_title("Image Height Distribution")
    axes[1].set_xlabel("Height (pixels)")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.show()
    
    # Print some additional statistics
    print("Image Resolution Statistics:")
    print(image_resolution_df.describe())


def brightness_and_contrast_analysis(images_train_path, images_val_path, images_test_path):
    """
    Analyzes the brightness and contrast of images across the train, val, and test datasets.
    
    Parameters:
        images_train_path (str): Path to the training images directory.
        images_val_path (str): Path to the validation images directory.
        images_test_path (str): Path to the testing images directory.
    
    Returns:
        dict: A dictionary containing brightness and contrast values for each dataset split.
    """
    all_paths = {
        "Train": images_train_path,
        "Validation": images_val_path,
        "Test": images_test_path,
    }

    brightness_values = {"Train": [], "Validation": [], "Test": []}
    contrast_values = {"Train": [], "Validation": [], "Test": []}

    def calculate_brightness_and_contrast(img):
        brightness = np.mean(img)
        contrast = np.std(img)
        return brightness, contrast

    for split_name, path in all_paths.items():
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        brightness, contrast = calculate_brightness_and_contrast(img)
                        brightness_values[split_name].append(brightness)
                        contrast_values[split_name].append(contrast)

    all_brightness = brightness_values["Train"] + brightness_values["Validation"] + brightness_values["Test"]
    all_contrast = contrast_values["Train"] + contrast_values["Validation"] + contrast_values["Test"]

    brightness_df = pd.Series(all_brightness)
    contrast_df = pd.Series(all_contrast)

    brightness_stats = brightness_df.describe()
    contrast_stats = contrast_df.describe()

    print("Brightness Analysis:")
    print(brightness_stats)
    print("\nContrast Analysis:")
    print(contrast_stats)

    return brightness_values, contrast_values


def plot_brightness_contrast(brightness_values, contrast_values):
    """
    Plots a scatter plot of brightness vs. contrast for the images in the datasets.
    
    Parameters:
        brightness_values (dict): Dictionary of brightness values for each dataset split.
        contrast_values (dict): Dictionary of contrast values for each dataset split.
    """
    brightness_values_flat = []
    contrast_values_flat = []

    for split_name in ["Train", "Validation", "Test"]:
        brightness_values_flat.extend(brightness_values[split_name])
        contrast_values_flat.extend(contrast_values[split_name])

    plt.figure(figsize=(10, 6))
    plt.scatter(brightness_values_flat, contrast_values_flat, alpha=0.6, color="blue")
    plt.xlabel("Brightness")
    plt.ylabel("Contrast")
    plt.title("Brightness and Contrast Analysis of Images")
    plt.grid(True)
    plt.show()


def color_analysis(data_dir: str, image_extensions=None):
    """
    Analyzes the average RGB color intensities of images in the given directory and its subfolders.
    
    Parameters:
        data_dir (str): The base directory containing images and subdirectories (train, val, test).
        image_extensions (list): List of allowed image file extensions (e.g., ['.jpg', '.jpeg', '.png']).
    
    Returns:
        None (Displays histograms of the RGB channels and prints the average intensities)
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    # Lists to store color channel means
    reds = []
    greens = []
    blues = []

    # Convert data_dir to a Path object
    data_dir = Path(data_dir)
    
    # Loop through the main directory and all subfolders
    for folder in data_dir.iterdir():
        if folder.is_dir(): 
            for filepath in folder.rglob("*"):
                if filepath.suffix.lower() in image_extensions:
                    try:
                        with Image.open(filepath) as img:
                            img = img.convert("RGB") 
                            img_array = np.array(img)
                            # Calculate mean for each RGB channel
                            reds.append(img_array[:, :, 0].mean())
                            greens.append(img_array[:, :, 1].mean())
                            blues.append(img_array[:, :, 2].mean())
                    except Exception as e:
                        print(f"Could not open image {filepath}: {e}")
    
    # Plot histograms for each color channel
    plt.figure(figsize=(15, 5))

    # Red channel histogram
    plt.subplot(1, 3, 1)
    plt.hist(reds, bins=30, color='red', edgecolor='black', alpha=0.7)
    plt.title("Red Channel Intensity")
    plt.xlabel("Average Intensity")
    plt.ylabel("Frequency")

    # Green channel histogram
    plt.subplot(1, 3, 2)
    plt.hist(greens, bins=30, color='green', edgecolor='black', alpha=0.7)
    plt.title("Green Channel Intensity")
    plt.xlabel("Average Intensity")
    plt.ylabel("Frequency")

    # Blue channel histogram
    plt.subplot(1, 3, 3)
    plt.hist(blues, bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Blue Channel Intensity")
    plt.xlabel("Average Intensity")
    plt.ylabel("Frequency")

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Show the plot
    plt.show()

    print("Average Red Intensity:", np.mean(reds))
    print("Average Green Intensity:", np.mean(greens))
    print("Average Blue Intensity:", np.mean(blues))