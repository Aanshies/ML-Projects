Step 1: Install Required Packages

!pip install nltk scikit-learn joblib

Step 2: Download the Model and Vectorizer Files

from google.colab import drive
drive.mount('/content/drive')

# Copy the files from Google Drive to the Colab environment
!cp /content/drive/MyDrive/path_to_your_files/sentiment_analysis_model.pkl /content/
!cp /content/drive/MyDrive/path_to_your_files/tfidf_vectorizer.pkl /content/

Step 3: Adapt the Script for Notebook Execution

import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Define text preprocessing functions
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatize words
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

def load_model_and_vectorizer():
    # Load the pre-trained model and vectorizer
    model = joblib.load('/content/sentiment_analysis_model.pkl')
    vectorizer = joblib.load('/content/tfidf_vectorizer.pkl')
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    # Preprocess the input text
    preprocessed_text = preprocess_text(text)
    # Transform the text using the TF-IDF vectorizer
    text_vectorized = vectorizer.transform([preprocessed_text])
    # Predict sentiment
    prediction = model.predict(text_vectorized)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Example text prediction
text_to_predict = "A wonderful little production.The filming technique is very unassuming- very old-time-BBC fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece."
result = predict_sentiment(text_to_predict, model, vectorizer)
print(f'Sentiment: {result}')


