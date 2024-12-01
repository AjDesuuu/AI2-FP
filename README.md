# AI2-FP
A repository for the AI2 final project requirement. This project is focused on using Yolo models for object detection using the BDD100k dataset.
Presentation can be found [here](https://www.canva.com/design/DAGXiwrcX5E/JC25ZTN9MRc8NrK4HMKqoA/edit?utm_content=DAGXiwrcX5E&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)!

Some instructions:
- You only need to download the FinalProject(NEW), the custom python files (.py) and the two .json files, picked_train and picked_val. Place all of those in files in one folde, all in the same level and the whole notebook will do its thing.
- The notebook can be run in local systems using jupyter notebook or vscode. It's also google colab compatible which saves your data in your google drive account.
- The BDD100k dataset will automatically be downloaded and extracted in the path specified after the conversion process.
- The subfolders needed as well as the dataset will automatically be created during run time. Please delete it after your runtime (if on local)
- Please change the annotations_train_path, annotations_val_path, output_labels_train_path and output_labels_val_path. Use absolute path if relative path causes issues for you as well
