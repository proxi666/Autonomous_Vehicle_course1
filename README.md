# Autonomous_Vehicle_course1
Project Title
This repository contains code implementations for various computer vision and machine learning tasks. Below is a brief description of each file included in this repository.

Files Overview
1. CNN.ipynb
This Jupyter Notebook contains a Convolutional Neural Network (CNN) implementation designed to classify images. The notebook walks through the process of building, training, and evaluating a CNN model on a given dataset. The steps include:

Data loading and preprocessing
Model architecture definition
Compilation and training of the model
Evaluation and visualization of the results
Usage: To run the notebook, simply open it in a Jupyter environment and execute the cells. Ensure all necessary libraries like TensorFlow or PyTorch are installed in your environment.

2. lanes.py
This Python script implements lane detection for autonomous driving applications. The script processes video input to detect lane lines using computer vision techniques such as edge detection and Hough line transformation. The key functionalities include:

Edge detection using the Canny edge detector
Region of interest selection to focus on the road lanes
Line detection using the Hough transform
Overlaying detected lines on the original video frames
Usage: To run this script, ensure you have OpenCV installed. The script is set up to read a video file, process it frame by frame, and save the output video with detected lanes highlighted.

3. mnist.ipynb
This Jupyter Notebook provides an implementation for recognizing handwritten digits from the MNIST dataset using a neural network. The notebook includes:

Data loading and exploration
Model creation using layers suitable for digit classification
Training the model and analyzing its performance
Visualizing the model’s predictions on the test set
Usage: Open the notebook in a Jupyter environment and execute the cells to train and evaluate the model on the MNIST dataset. Required libraries like Keras or TensorFlow should be installed prior to running the notebook.
