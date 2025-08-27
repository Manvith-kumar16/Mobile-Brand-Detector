Mobile Detector Model Training
This directory contains the script (train.py) used to train the machine learning model. The model is a MobileNetV2 pre-trained on the ImageNet dataset, which is then fine-tuned for the specific task of mobile brand and model detection using transfer learning.

How to Run
Install Python Dependencies: Ensure you have the necessary libraries installed.

pip install tensorflow numpy Pillow

Organize Your Dataset: The script expects your data to be organized in a specific way. The root directory should contain sub-directories, with each sub-directory named after a brand and model (e.g., Apple_iPhone15).

Update the Path: Change the DATASET_PATH variable in train.py to point to your dataset directory.

Run the Script:

python train.py

The script will train the model, showing progress in the console, and save the final mobile_detector_model.h5 and class_labels.txt files in the same directory.