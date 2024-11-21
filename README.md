# BankNotesFraudDetection
This project uses machine learning techniques to detect fraudulent banknotes based on various features extracted from images of the notes. The dataset contains data on banknotes labeled as either genuine or fraudulent.
📋 Table of Contents

    Project Overview
    Technologies Used
    Dataset
    Model Approach
    Installation
    Usage
    Results
    Contributing
    License

📖 Project Overview

The goal of this project is to predict whether a banknote is genuine or fraudulent based on its physical attributes such as the length, width, and other characteristics. This project demonstrates the power of machine learning in classifying data based on numerical features.
💻 Technologies Used

    Python 3.12
    Libraries and Frameworks:
        Pandas
        Scikit-learn
        Matplotlib
        NumPy
        Seaborn

📂 Dataset

The dataset used in this project is the Bank Note Authentication Dataset from UCI Machine Learning Repository. It contains 1,372 instances with 4 features:

    Variance of the wavelet transformed image of the banknote
    Skewness of the wavelet transformed image of the banknote
    Curtosis of the wavelet transformed image of the banknote
    Entropy of the wavelet transformed image of the banknote

Dataset Columns:

    Variance
    Skewness
    Curtosis
    Entropy
    Class: 0 for genuine, 1 for fraudulent

🧠 Model Approach
1. Data Preprocessing

    Data Cleaning: Handled missing values (if any).
    Feature Scaling: Applied Min-Max Scaling or Standardization to ensure the features are on the same scale.
    Train-Test Split: Split the dataset into training and testing sets (typically 80-20 split).

2. Model Selection

We experimented with various machine learning models to classify the data:

    Logistic Regression
    Random Forest Classifier
    Support Vector Machine (SVM)
    K-Nearest Neighbors (KNN)
    Naive Bayes

3. Model Evaluation

The models were evaluated using:

    Accuracy
    Precision
    Recall
    F1-Score

We also used cross-validation to ensure the robustness of our models.
🛠 Installation

    Clone the repository:

git clone https://github.com/yourusername/BankNoteFraudDetection.git  
cd BankNoteFraudDetection  

Install dependencies:

    pip install -r requirements.txt  

    Download the dataset and place it in the data/ folder. You can also use the dataset from the UCI repository.

🚀 Usage

    Train the Model:
    Train a model using the train.py script:

python train.py  

Evaluate the Model:
Evaluate the trained model’s performance on the test dataset:

python evaluate.py  

Run the Prediction:
Use the trained model to predict whether a given banknote is genuine or fraudulent:

    python predict.py  

    Interactive Notebook:
    Use the BankNoteFraudDetection.ipynb Jupyter Notebook for an interactive approach to training, testing, and visualizing the results.

📊 Results

The best performing model achieved the following results:
Model	Accuracy	Precision	Recall	F1-Score
Random Forest Classifier	99.0%	0.98	0.99	0.99


🤝 Contributing

Contributions are welcome! Follow these steps:

    Fork the repository.
    Create a new branch: git checkout -b feature-name.
    Commit your changes: git commit -m 'Add feature'.
    Push to the branch: git push origin feature-name.
    Submit a pull request.

📝 License

This project is licensed under the MIT License.
