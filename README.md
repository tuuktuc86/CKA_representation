Neural Network Representation Similarity - CKA Experiment

This project is an experimental implementation of Centered Kernel Alignment (CKA), as discussed in the paper: Similarity of Neural Network Representation Revisited.

📌 Project Overview

This project explores the CKA metric to analyze neural network representations.

The source code is based on centered-kernel-alignment but with modifications to certain functions.

The actual CKA implementation used in this project is located inside the centered-kernel-alignment/ folder.



1️⃣ Training Scripts (``********************):

These scripts train different models on the MNIST dataset. The models include:

Linear Model

Convolutional Neural Network (CNN)

Equivariant CNN

Each script trains a model and saves the trained weights for further evaluation.

2️⃣ Testing Scripts (test_*.py********************************):

These scripts validate the trained models' performance on the MNIST dataset. They load the trained weights and compute accuracy metrics.

3️⃣ CKA Comparison Scripts (compare_*.py********************************):

These scripts calculate the CKA similarity between different models.

The file names and structure follow an intuitive pattern, making it easy to understand their purpose.

For better readability, the scripts include detailed comments (though written in Korean—sorry! 😅).

📜 Notes

The centered-kernel-alignment/ folder contains the actual implementation of CKA.

