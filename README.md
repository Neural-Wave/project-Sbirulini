# Neural Wave - Duferco
youtube video: https://youtu.be/SbFbltCm8mM
## Group Members - Sbirulini

- [Mattia Gianinazzi](mailto:mattia.gianinazzi.1@usi.ch)
- [Volodymyr Karpenko](mailto:volodymyr.karpenko@usi.ch)
- [Marzio Lunghi](mailto:marzio.lunghi@gmail.com)
- [Alessandro De Grandi](mailto:degraa@usi.ch)
- [Qianbo Zang](mailto:qianbo.zang@uni.lu)

## Project Description

Align mental bars with the help of AI. (Thanks for the dataset without the labels)

## Overview

To address the challenge of labeling and classifying a large set of images, we developed a custom labeling tool, implemented a training pipeline for the ResNet18 model, and created a web application to provide online and real-time predictions.

### Key Components:
- **Manual Labeling Tool:** `labeler.py` allows for the manual labeling of images.
- **Model Training:** 'model_training.ipynb` details the finetuning process of a ResNet18 model using PyTorch Lightning.
- **Web Application:** A Flask-based application with drag-and-drop functionality for online predictions, as well as real-time streaming capabilities.

## Features

1. **Custom Labeling Tool:** Simplifies the manual labeling process, aiding in the annotation of XXXX images.
2. **ResNet18 Finetuning with PyTorch Lightning:** Enables effective model training and fine-tuning.
3. **Interactive Web Application:**
   - **Drag and Drop Upload:** Easily upload images for immediate predictions.
   - **Real-time Predictions with Grad-CAM Heatmaps:** Displays model focus areas, highlighting regions that influence predictions.

![Image Streaming 1](imageStreaming0.png)
![Image Streaming 1](imageStreaming1.png)
![Image Streaming 2](imageStreaming2.png)

## Directory Structure for training/testing
To train the model, the images need to be in "data/train_set/" with subdirectories "aligned" and "not_aligned", the labels should be in a .json file located in the root folder with the following structure:

{filename: label, ...}
es:
{"img_00100.jpg": "not_aligned", "img_00101.jpg": "aligned",  ... }.

To test the model, the test set is expected in the "data/example_set/" with subdirectories "aligned" and "not_aligned".

For the web app, server.py expects a file named model.pth in the root folder, and a images in "/data/video/"

## Installation
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt

## Running
python server.py to start the server on localhost:5000 with the drag/drop feature, navigate to localhost:5000/stream for the real-time streaming 

## labeling effort (total 10_000)

- Mattia Gianinazzi, start at: 8000, currently at : 9340,
- Volodymyr Karpenko,start at: 6000, currently at: 7257,
- Marzio Lunghi,start at: 4000, currently at: 5625,
- Alessandro De Grandi,start at: 2000, currently at: 2300,
- Qianbo Zang,start at: 0, currently at: y,



