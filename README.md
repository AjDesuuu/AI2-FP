# AI2-FP
A repository for the AI2 final project requirement. This project is focused on using Yolo models for object detection using the BDD100k dataset.
Presentation can be found [here](https://www.canva.com/design/DAGXiwrcX5E/JC25ZTN9MRc8NrK4HMKqoA/edit?utm_content=DAGXiwrcX5E&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)!

Some instructions:
- The BDD100k dataset will automatically be downloaded and extracted in the path specified after the conversion process.
- You need to create a "dataset" folder with the folder structure specified in the notebook. Particularly this part ![File Structure](file%20structure.png)
- For the train dataset, please limit the images to 2,000 samples for faster training time.
- For the validation dataset, please limit the images to 1,500 samples.
- You can easily limit the images by moving it into a subfolder. within the train, test and val subfolders.
- Please change the annotations_train_path, annotations_val_path, output_labels_train_path and output_labels_val_path. Use absolute path if relative path causes issues for you as well.
