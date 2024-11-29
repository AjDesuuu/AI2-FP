
import random
from tqdm import tqdm
import os
import pandas as pd
import json
import shutil
import yaml


def flatten_folders(source_dir, destination_dir):
    """
    Recursively moves all files from subdirectories of the source directory to the destination directory.
    The function flattens the folder structure, ensuring all files are in the root of the destination directory.

    Parameters:
        source_dir (str): Path to the directory containing files and subdirectories to flatten.
        destination_dir (str): Path to the directory where all files should be moved.

    Returns:
        None
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Collect all file paths in the source directory and subdirectories
    files_to_move = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            files_to_move.append(os.path.join(root, file))

    # If no files are found, inform the user and exit
    if not files_to_move:
        print(f"No files found in '{source_dir}'.")
        return

    # Move each file to the destination directory with a progress bar
    for file_path in tqdm(files_to_move, desc="Flattening folders"):
        dest_path = os.path.join(destination_dir, os.path.basename(file_path))
        shutil.move(file_path, dest_path)

    print(f"Moved {len(files_to_move)} files from '{source_dir}' to '{destination_dir}'.")

def copy_files(source_dir, destination_dir, items=None, pick=None):
    """
    Copies files from the source directory to the destination directory.

    - If `items` is specified, checks if the destination already has sufficient files 
      and only copies additional files if needed.
    - If `pick` is specified, uses a JSON file containing a list of filenames to copy.
    - If neither is specified, copies all files.

    Parameters:
        source_dir (str): Path to the source directory.
        destination_dir (str): Path to the destination directory.
        items (int, optional): Number of files to copy. Ignored if `pick` is specified.
        pick (str, optional): Path to a JSON file containing a list of filenames to copy.

    Returns:
        None
    """
    os.makedirs(destination_dir, exist_ok=True)

    # Read the list of all files in the source directory
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]
    if not all_files:
        print(f"No files found in '{source_dir}'.")
        return

    # Get the list of existing files in the destination directory
    existing_files = [f for f in os.listdir(destination_dir) if os.path.isfile(os.path.join(destination_dir, f))]
    existing_count = len(existing_files)

    # Use the "pick" JSON file to select specific files
    if pick:
        try:
            with open(pick, 'r') as file:
                picked_files = json.load(file)
            files_to_copy = [f for f in all_files if f in picked_files]
        except Exception as e:
            print(f"Error reading 'pick' file: {e}")
            return
    # Otherwise, randomly sample a subset of files or copy all files
    elif items is not None:
        remaining_needed = max(0, items - existing_count)
        if remaining_needed == 0:
            print(f"Destination already contains {existing_count} files, which meets or exceeds the requested {items}.")
            return
        files_to_copy = random.sample(all_files, min(remaining_needed, len(all_files)))
    else:
        files_to_copy = all_files

    skipped_count = 0

    # Copy the files to the destination directory
    for file in tqdm(files_to_copy, desc="Copying files"):
        src_path = os.path.join(source_dir, file)
        dest_path = os.path.join(destination_dir, file)

        # Check if the file already exists
        if os.path.exists(dest_path):
            skipped_count += 1
            continue

        # Copy the file
        shutil.copy(src_path, dest_path)

    print(f"Copying complete. Skipped {skipped_count} files due to existing duplicates.")
    print(f"Copied {len(files_to_copy) - skipped_count} new files to '{destination_dir}'.")



def move_files(source_dir, destination_dir):
    """
    Recursively moves files and subdirectories from the source directory to the destination directory,
    ensuring no duplicate moves are performed.

    Parameters:
        source_dir (str): Path to the directory containing the files and subdirectories to move.
        destination_dir (str): Path to the directory where files and subdirectories should be moved.

    Returns:
        None
    """
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # List files and subdirectories in source and destination
    source_items = set(os.listdir(source_dir))  # files and subdirectories in source
    destination_items = set(os.listdir(destination_dir))  # items already in destination

    # Identify items to move (files or subdirectories)
    items_to_move = source_items - destination_items

    if not items_to_move:
        print("No files or subdirectories to move. All are already in the destination.")
        return

    # Function to move a file or directory
    def move_item(src, dest):
        if os.path.isdir(src):  # If it's a directory, use shutil.move for the whole folder
            shutil.move(src, dest)
        else:  # If it's a file, just move the file
            shutil.move(src, dest)

    # Move items (both files and subdirectories) with a progress bar
    for item in tqdm(items_to_move, desc="Moving items"):
        source_path = os.path.join(source_dir, item)
        destination_path = os.path.join(destination_dir, item)
        
        move_item(source_path, destination_path)

    print(f"Moved {len(items_to_move)} item(s) from {source_dir} to {destination_dir}.")
    

def convert_annotations_to_yolo(annotations_train_path, annotations_val_path, 
                                 output_labels_train_path, output_labels_val_path, 
                                 image_width, image_height, class_mapping):
    """
    Converts JSON annotations to YOLO format for both train and validation datasets.

    Parameters:
        annotations_train_path (str): Path to the JSON annotations for training.
        annotations_val_path (str): Path to the JSON annotations for validation.
        output_labels_train_path (str): Directory to save YOLO format labels for training.
        output_labels_val_path (str): Directory to save YOLO format labels for validation.
        image_width (int): Width of the original images (pixels).
        image_height (int): Height of the original images (pixels).
        class_mapping (dict): Dictionary mapping class names to class IDs.

    Returns:
        None: Saves the YOLO labels in the specified directories.
    """
    # Ensure output directories exist
    os.makedirs(output_labels_train_path, exist_ok=True)
    os.makedirs(output_labels_val_path, exist_ok=True)
    
    def convert_to_yolo_format(annotations, output_dir):
        """
        Internal function to handle the conversion of individual annotations.
        """
        for annotation in annotations:
            image_name = annotation["name"].split(".")[0]
            label_file_path = os.path.join(output_dir, f"{image_name}.txt")

            # Check if label file already exists; skip if it does
            if os.path.exists(label_file_path):
                continue

            yolo_labels = []
            for obj in annotation["labels"]:
                category = obj["category"]
                if category not in class_mapping:
                    continue  # Skip classes not in our list

                class_id = class_mapping[category]
                box2d = obj["box2d"]

                # Calculate YOLO format values
                x_center = ((box2d["x1"] + box2d["x2"]) / 2) / image_width
                y_center = ((box2d["y1"] + box2d["y2"]) / 2) / image_height
                width = (box2d["x2"] - box2d["x1"]) / image_width
                height = (box2d["y2"] - box2d["y1"]) / image_height

                yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

            # Save the YOLO labels in the output directory
            with open(label_file_path, "w") as label_file:
                label_file.write("\n".join(yolo_labels))

    # Convert train annotations
    with open(annotations_train_path, 'r') as f:
        train_annotations = json.load(f)
    print("Processing training annotations...")
    convert_to_yolo_format(train_annotations, output_labels_train_path)

    # Convert validation annotations
    with open(annotations_val_path, 'r') as f:
        val_annotations = json.load(f)
    print("Processing validation annotations...")
    convert_to_yolo_format(val_annotations, output_labels_val_path)

    print("Conversion complete! YOLO labels are saved in the specified output directories.")


def yolo_labels_to_dataframe(output_dir, img_width, img_height):
    """
    Reads YOLO label files from a directory and converts them into a DataFrame.

    Parameters:
        output_dir (str): Path to the directory containing YOLO label files.
        img_width (int): Width of the original image (pixels).
        img_height (int): Height of the original image (pixels).

    Returns:
        DataFrame: Parsed YOLO labels including filename, class ID, and bounding box details.
    """
    label_data = []

    # List all .txt files in the directory
    label_files = [f for f in os.listdir(output_dir) if f.endswith(".txt")]

    # Iterate through the label files with a progress bar
    for label_file in tqdm(label_files, desc="Processing YOLO labels"):
        file_path = os.path.join(output_dir, label_file)
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Convert YOLO normalized values to pixel values
                x_center_px = x_center * img_width
                y_center_px = y_center * img_height
                width_px = width * img_width
                height_px = height * img_height

                # Calculate additional metrics
                area_px = width_px * height_px
                aspect_ratio = width_px / height_px if height_px > 0 else None

                # Add to list
                label_data.append({
                    "filename": label_file.replace(".txt", ".jpg"),  # Match with the corresponding image filename
                    "class_id": class_id,
                    "x_center": x_center_px,
                    "y_center": y_center_px,
                    "bbox_width": width_px,
                    "bbox_height": height_px,
                    "area": area_px,
                    "aspect_ratio": aspect_ratio
                })

    # Convert list to DataFrame
    return pd.DataFrame(label_data)



def create_dataset_yaml(dataset_name, base_path, output_dir, class_mapping):
    """
    Creates a dataset YAML file dynamically.

    Parameters:
        dataset_name (str): Name of the dataset (e.g., 'dataset1', 'dataset2').
        base_path (str): Path to the root folder where the datasets are stored.
        output_dir (str): Directory to save the generated YAML file.
        class_mapping (dict): Dictionary mapping class IDs to class names.

    Returns:
        str: Path to the created YAML file.
    """
 
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct YAML content
    yaml_content = {
        "path": os.path.abspath(base_path).replace("\\", "/"),
        "train": dataset_name + "/images/train",
        "val": dataset_name + "/images/val",
        "nc": len(class_mapping),  # Number of classes
        "names": {k: class_mapping[k] for k in class_mapping},  # Keep order as provided
    }

    # Create the YAML file
    yaml_path = os.path.join(output_dir, f"{dataset_name}.yaml")
    with open(yaml_path, "w") as file:
        yaml.dump(
            yaml_content,
            file,
            default_flow_style=False,
            sort_keys=False,  # Prevent sorting of dictionary keys
        )

    print(f"YAML file created at: {yaml_path}")
    return yaml_path



def save_image_names_to_json(source_dir, output_json):
    """
    Scans a folder for image files and saves their filenames to a JSON file.

    Parameters:
        source_dir (str): Path to the folder containing images.
        output_json (str): Path to the JSON file where the list of image names will be saved.

    Returns:
        None
    """
    # Define supported image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    
    # Get a list of all image files in the source directory
    image_files = [f for f in os.listdir(source_dir) 
                   if os.path.isfile(os.path.join(source_dir, f)) and 
                   os.path.splitext(f)[1].lower() in image_extensions]
    
    # Save the list to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(image_files, json_file, indent=4)
    
    print(f"Saved {len(image_files)} image filenames to '{output_json}'.")


