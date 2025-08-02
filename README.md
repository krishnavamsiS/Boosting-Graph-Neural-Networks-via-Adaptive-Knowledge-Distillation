Boosting Graph Neural Networks via Adaptive Knowledge Distillation

This repository contains the implementation of a framework to enhance Graph Neural Network (GNN) performance using Adaptive Knowledge Distillation and Weight Augmentation. The approach optimizes GNNs for node classification and link prediction tasks, improving their accuracy and efficiency.

Overview

The project leverages a teacher-student model architecture where a complex teacher GNN transfers knowledge to a simpler student GNN through adaptive knowledge distillation. Weight augmentation is applied to enhance model robustness, making it suitable for various graph-based tasks.

Features

Pre-trained models for node classification and link prediction.

Configurable scripts to train student and teacher models.

Support for customizing model architectures and training parameters.

Usage

Running the Student Model:

Execute student.sh to train the student model.

Modify student.sh to change:

Task (node classification or link prediction).

Student and teacher model architectures.

Training hyperparameters.

Retraining the Teacher Model:

Run teacher.sh to retrain the teacher model if needed.

Dependencies:

Ensure required libraries (e.g., PyTorch, PyTorch Geometric) are installed. Refer to requirements.txt for details.
