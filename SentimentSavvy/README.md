# SentimentSavvy: NLP for Sentiment Analysis

**SentimentSavvy** is a machine learning project that focuses on sentiment analysis of text data, specifically for classifying text into positive and negative sentiments. The model is developed in Python using natural language processing (NLP) techniques and achieves 85% accuracy on test data using a logistic regression algorithm.

## Table of Contents

- Project Overview
- Features
- Installation
- Dataset
- Usage
  - Running Predictions
  - Training the Model
- Model Details
  - Text Preprocessing
  - TF-IDF Vectorization
  - Model Training
  - Model Evaluation
- Google Colab Notebook
- Contributing
- License
  
## Project Overview

SentimentSavvy is a sentiment analysis model that classifies text data into positive or negative sentiments. The project leverages natural language processing techniques, such as text preprocessing and TF-IDF vectorization, to convert text data into numerical features. The core of the model is a logistic regression algorithm that has been trained and evaluated to achieve high accuracy.

## Features

- **Text Preprocessing:** Implements noise removal, lowercasing, and lemmatization.
- **TF-IDF Vectorization:** Converts text data into numerical features using TF-IDF.
- **Sentiment Classification:** Classifies text as positive or negative using logistic regression.
- **Model Evaluation:** Includes precision, recall, F1-score, and ROC-AUC metrics.

## Installation

### 1. Clone the Repository

Clone the repository to your local machine using the following command:

```bash
https://github.com/Aanshies/ML-Projects/tree/main/SentimentSavvy
```

### 2. Install Dependencies

Navigate to the project directory and install the required Python packages using pip:

```bash
cd ml-projects/SentimentSavvy
pip install -r requirements.txt
```

### 3. Dataset

The dataset used for training the model is included in the `data/` directory.

- **File:** `IMDB_Dataset.csv`
- **Size:** (Include file size)
- **Description:** Contains movie reviews and corresponding sentiment labels.

## Usage

### Running Predictions

To make predictions on new text data using the pre-trained model, run the `predict.py` script:

```bash
python src/predict.py "Your text to analyze"
```

### Training the Model

If you want to retrain the model, use the `notebooks/SentimentSavvy_Model.ipynb` Colab notebook or run the code in Jupyter Notebook. The notebook is pre-configured to use the dataset provided in the `data/` directory.

## Model Details

### Text Preprocessing

The text data is preprocessed using the following techniques:

- **Lowercasing:** Converts all text to lowercase to ensure uniformity.
- **Noise Removal:** Removes special characters, HTML tags, and extra spaces.
- **Stop Words Removal:** Removes common English stop words.
- **Lemmatization:** Reduces words to their base form.

### TF-IDF Vectorization

The preprocessed text data is transformed into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique. This helps in representing the importance of each word in the context of the dataset.

### Model Training

The model is trained using a logistic regression algorithm. The training process is executed in the `notebooks/SentimentSavvy_Model.ipynb` notebook.

- **Algorithm:** Logistic Regression
- **Accuracy:** 85% on test data
- **Pre-trained Model:** Saved as `sentiment_analysis_model.pkl`
- **Vectorizer:** Saved as `tfidf_vectorizer.pkl`

### Model Evaluation

The model's performance is evaluated using the following metrics:

- **Accuracy:** Measures the percentage of correct predictions.
- **Precision:** Measures the proportion of true positive results among all positive predictions.
- **Recall:** Measures the proportion of true positive results among all actual positives.
- **F1-score:** The harmonic mean of precision and recall.
- **ROC-AUC:** Measures the area under the receiver operating characteristic curve.

## Google Colab Notebook

The training process and code are also available in a Google Colab notebook. You can use the notebook to modify, retrain, and experiment with the model.

- [Open Colab Notebook](https://colab.research.google.com/drive/your-colab-link)

## Contributing

Contributions to SentimentSavvy are welcome! If you would like to contribute:

1. **Fork the repository** on GitHub.
2. **Clone your fork** to your local machine.
3. **Create a new branch** for your feature or bugfix.
4. **Commit your changes** with a descriptive message.
5. **Push your branch** to your forked repository.
6. **Create a pull request** to the main repository with a description of your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

