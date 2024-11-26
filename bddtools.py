
from tqdm import tqdm
import os
import pandas as pd
import json

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
