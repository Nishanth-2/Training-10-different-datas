Animals-10 Image Classification
This repository contains a Convolutional Neural Network (CNN) built using TensorFlow and Keras to classify images from the Animals-10 dataset. The dataset consists of 10 categories of animals, available here on Kaggle.

Dataset
Source: Animals-10 Dataset on Kaggle
Classes: 10 different animal categories.
Format: Images organized in subdirectories.
Project Structure
train.py: Code to train the model using transfer learning (VGG16).
test.py: Code to test the model on new images.
README.md: Documentation of the project.
Getting Started
Download the dataset from Kaggle.
Clone this repository.
Run the train.py script to train the model.
Use test.py for prediction on new images.
Model
Base Model: VGG16 (pre-trained on ImageNet).
Custom Layers: Dense and Dropout layers for classification.
Requirements
Python 3.x
TensorFlow 2.x
Keras
NumPy

pip install -r requirements.txt
Usage
Train the model:

python train.py
Test the model:

python test.py --image_path /path/to/image
