# Hand Gesture Recognition with Deep Learning

This project aims to implement a hand gesture recognition system using deep learning techniques. It focuses on classifying hand gestures based on a dataset of images representing various hand gestures. The model uses a neural network architecture built with PyTorch, trained to accurately classify gestures for real-time applications.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Testing](#testing)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project is designed to recognize hand gestures using a deep learning model built with PyTorch. It leverages computer vision techniques and neural networks to classify hand gestures into predefined categories. The system can be integrated into various real-time applications such as controlling devices via hand gestures.

## Features

- **Dataset Loading**: Custom dataset loader for handling CSV files containing hand gesture data.
- **Deep Learning Model**: A neural network built with fully connected layers, including ReLU activation functions, dropout layers, and batch normalization.
- **Training & Evaluation**: Training using cross-entropy loss with early stopping to prevent overfitting.
- **Confusion Matrix**: Generated confusion matrix to evaluate model performance.
- **Classification Report**: Generated classification report with precision, recall, and F1-score.
- **GPU Acceleration**: Training and testing on GPU (if available) for faster performance.

## Installation

Follow these steps to set up the project on your local machine:

### Clone the repository

```bash
git clone --filter=blob:none --no-checkout https://github.com/NhatTran-97/Deep_Learning_Projects.git
cd hand-gesture-recognition
