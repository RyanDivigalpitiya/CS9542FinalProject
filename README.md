# Detecting Pneumonia in Lung X-Ray Images using Convolutional Neural Networks

### Computer Science 9542 Advanced Artificial Intelligence II, Western University

### Contributors: *Ryan Divigalpitiya, Heather Hartley, Mohmmd Alanbary*

> **Note:** This project was submitted for grading, May 7, 2021

### Description

With the rise of Covid-19, the AI community has been sought after to develop detection techniques that will help in detecting pneumonia along with being able to discriminate between the different causes of pneumonia, such as Covid-19.

### Dataset

- Our dataset for this project consists of lung x-ray scans of both healthy and unhealthy patients; unhealthy patients have been diagnosed with either a viral-based pneumonia infection or a bacterial-based pneumonia infection. Pneumonia is an infection in the lungs that causes inflammation. Its causes can be either bacterial-based or viral-based. Covid-19, a virus, causes pneumonia in more severe cases.
- For the patients that had viral-induced pneumonia in our dataset, the dataset also includes whether they received a positive or negative diagnosis for Covid-19. In other words, their pneumonia was caused by the coronavirus if positive.
- If their diagnosis is negative, then that means the patient has some other viral chest infection that is not Covid-19.

### Techniques

- We will be using computer vision techniques, specifically, convolutional neural networks (CNN), to help classify between healthy and unhealthy patients (ie. if patients have pneumonia or not). We will also attempt to build a CNN model that can differentiate between different types of pneumonia infections, including being able to detect Covid-19 specifically.
- Pre-processing techniques may be explored to help our CNN model learn better from the images. This may include cropping and/or enhancing features of the image we deem will benefit the performance of the model. We also plan on experimenting with different neural network architectures in order to achieve the most optimal performance for our model.

