# Dog Vision Project

## Overview

This project utilizes deep learning and transfer learning to classify dog breeds based on images. It is developed using **TensorFlow**, **Keras**, and **TensorFlow Hub**, with a pre-trained MobileNet model fine-tuned for the specific task of dog breed classification. The project demonstrates how to leverage existing models for efficient training and deployment in a real-world computer vision problem.

This project was part of the **Complete A.I. & Machine Learning, Data Science Bootcamp** by **Zero To Mastery Academy**. It serves as a practical application of machine learning concepts, showcasing techniques in deep learning, transfer learning, and model evaluation.

---

## Key Features
- **Transfer Learning**: The model uses the pre-trained ResNet50 architecture from **TensorFlow Hub** and fine-tunes it for dog breed classification.
- **High Accuracy**: Achieved 90% training accuracy and 85% validation accuracy.
- **Python Libraries**: Developed with **TensorFlow**, **Keras**, and **TensorFlow Hub** for efficient model training and deployment.

---

## Data

Unfortunately, due to the size of the dataset, it is not uploaded in this repository. However, the dataset used is the **Kaggle Dog Breed Dataset**, which contains labeled images of various dog breeds. You can download it from [here](https://www.kaggle.com/competitions/dog-breed-identification).

### Instructions for Data Setup
1. Download the dataset from the link above.
2. Extract the images into a folder named `data/`.
3. The project expects the following directory structure: data/ ├── images/ └── labels/
4. Place the dataset in the `data/` directory within this repository.

---

## Setup Instructions

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Matplotlib, NumPy (for image processing and visualization)

### Installing Dependencies

1. Clone the repository:
   git clone https://github.com/Nosckid/dog-vision.git
2. Navigate to the project directory:
   cd dog-vision
3. **Note:** If you don't have the dataset already, follow the instructions above to download it and place it in the `data/` folder.

### How to Run the Project
1. Open the notebook:
   jupyter notebook dog-vision.ipynb
2. Run the cells in the notebook to:
   - Preprocess the data
   - Load the pre-trained ResNet50 model
   - Fine-tune the model for dog breed classification
   - Evaluate the model's performance
3. After training, you can visualize the model’s predictions and performance metrics directly in the notebook.

### Model Architecture and Results
This project uses the **MobileNet** architecture for transfer learning. The pre-trained model is fine-tuned with the dog breed dataset to improve its classification accuracy.

#### Results
- Training accuracy: 90%
- Validation accuracy: 85%
- Loss: 0.2

#### Model Evaluation
- The model was evaluated on a test dataset, achieving 85% accuracy in classifying various dog breeds.
- Metrics such as precision, recall, and F1 score were computed for model performance evaluation (you can explore these in the notebook).


### Acknowledgement
- Special thanks to Zero To Mastery Academy for the Complete A.I. & Machine Learning, Data Science Bootcamp, which provided the foundational knowledge for this project.
