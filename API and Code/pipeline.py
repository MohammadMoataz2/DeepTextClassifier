from keras_preprocessing.text import tokenizer_from_json
import json
from nltk.stem import PorterStemmer
import re
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np





label_map = {
    0: 'Politics',
    1: 'Sport',
    2: 'Technology',
    3: 'Entertainment',
    4: 'Business'
}



with open('tokenizer.json') as f:
    data = json.load(f)
    tokenizer = tokenizer_from_json(data)



model = load_model('best_model.h5')



def clean_text(text):
    text = text.lower() 
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  
    
    tokens = word_tokenize(text) 
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens if word not in stopwords.words('english')]  
    
    return ' '.join(stemmed_tokens)

def classify_text(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=150)
    prediction = model.predict(padded_sequence)
    predicted_label = np.argmax(prediction)
    return label_map[predicted_label]









