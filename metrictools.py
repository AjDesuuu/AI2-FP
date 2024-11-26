import pandas as pd
import matplotlib.pyplot as plt

def plot_yolo_metrics(results_csv_path):
    """
    Plot training and validation metrics from a YOLOv8 results file.

    Parameters:
        results_csv_path (str): Path to the 'results.csv' file generated by YOLOv8 training.
    """
    try:
        # Load the data
        data = pd.read_csv(results_csv_path)

        # Extract metrics
        epochs = data['epoch']
        
        # Calculate total losses for training and validation
        train_loss = data['train/box_loss'] + data['train/cls_loss'] + data['train/dfl_loss']
        val_loss = data['val/box_loss'] + data['val/cls_loss'] + data['val/dfl_loss']
        
        # Extract accuracy metrics
        precision = data['metrics/precision(B)']
        recall = data['metrics/recall(B)']
        mAP50 = data['metrics/mAP50(B)']
        mAP50_95 = data['metrics/mAP50-95(B)']

        # Plot the data
        plt.figure(figsize=(14, 8))

        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_loss, label='Training Loss', color='blue')
        plt.plot(epochs, val_loss, label='Validation Loss', color='orange', linestyle='--')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

        # Plot accuracy metrics
        plt.subplot(2, 1, 2)
        plt.plot(epochs, precision, label='Precision', color='green')
        plt.plot(epochs, recall, label='Recall', color='red')
        plt.plot(epochs, mAP50, label='mAP@50', color='purple')
        plt.plot(epochs, mAP50_95, label='mAP@50-95', color='brown')
        plt.xlabel('Epochs')
        plt.ylabel('Metrics')
        plt.title('Validation Metrics')
        plt.legend()

        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {results_csv_path}")
    except KeyError as e:
        print(f"Missing column in results file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


