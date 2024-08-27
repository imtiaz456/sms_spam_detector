import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():   # if i is alpha numeric
            y.append(i)
    text = y[:]  # its mean cloning..
    y.clear()
    
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl','rb'))  # rb mean read binary mode
model = pickle.load(open('model.pkl','rb'))

st.title('Email/SMS Spam Classifier')

input_sms = st.text_input("Enter the message")

if st.button('Predict'):

    # Now we have to perform 3 processes
    # 1. Preprocessing
    transformed_sms =transform_text(input_sms)

    # 2. Vectorization
    vector_input = tfidf.transform([transformed_sms]) # we will pass transformed_sms in a list

    # 3. Prediction
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")