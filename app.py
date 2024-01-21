# Core Packages
import streamlit as st
import altair as alt
import plotly.express as px
import tensorflow as tf
# EDA Packages
import pandas as pd
import numpy as np
from datetime import datetime
import joblib 
data_1 = pd.read_json("./data/Sarcasm_Headlines_Dataset.json", lines=True)
data_2 = pd.read_json("./data/Sarcasm_Headlines_Dataset_v2.json", lines=True)
data =  pd.concat([data_1, data_2])

pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl","rb"))
# Load Model sentiment analysis 
import joblib 
pipe_sentiment = joblib.load(open("./models/sentiments_classifier_pipe_lr.pkl","rb"))


# Chargement du modèle
model = tf.keras.models.load_model('./models/detect sarcasms_eng.h5')

from textblob import TextBlob
import nltk
nltk.download('punkt')
import nltk
nltk.download('stopwords')

# Track Utils
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table

# Function
def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def predict_sentiment(docx):
    # Using TextBlob for sentiment analysis
    blob = TextBlob(docx)
    sentiment = blob.sentiment

    # Mapping sentiment polarity to positive, negative, or neutral
    if sentiment.polarity > 0:
        return "positive"
    elif sentiment.polarity < 0:
        return "negative"
    else:
        return "neutral"



def get_sentiment_prediction_proba(docx):
    # TextBlob doesn't provide probability scores, so we return an empty list
    return []
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
def clean_text(text):
    text = text.lower()
    
    pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    emoji = re.compile("["
                           u"\U0001F600-\U0001FFFF"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    
    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)        
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text) 
    text = re.sub(r"\'ll", " will", text)  
    text = re.sub(r"\'ve", " have", text)  
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def CleanTokenize(df):
    head_lines = list()
    lines = df["headline"].values.tolist()

    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        head_lines.append(words)
    return head_lines

head_lines = CleanTokenize(data)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
max_length = 25
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(head_lines)
sequences = tokenizer_obj.texts_to_sequences(head_lines)

def predict_sarcasm(s):
    x_final = pd.DataFrame({"headline":[s]})
    test_lines = CleanTokenize(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)
    
    pred*=100
    if pred[0][0] >= 50:
       return "It's a sarcasm!"
    else:
       return "It's not a sarcasm."
    
emotions_emoji_dict = {"anger" : "😠", "disgust" : "🤮", "fear" : "😨😱", "happy" : "🤗", "joy" : "😂", "neutral" : "😐", "sad" : "😔", "sadness" : "😔", "shame" : "😳", "surprise" : "😮"}

# Main Application
def main():
    

    menu = ["🏠 Home", "📈 Monitor", " ℹ️ About","sarcasm"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    if choice == "🏠 Home":
        st.title("Emotion and Sentiment Analysis App")
        add_page_visited_details("Home", datetime.now())
        st.subheader("Text Analysis")

        # Add tabs for different text analysis tasks
        selected_tab = st.radio("Select Text Analysis Task", ["Emotion Detection", "Sentiment Analysis"])

        with st.form(key='text_analysis_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)

            # Apply the selected text analysis function
            if selected_tab == "Emotion Detection":
                prediction = predict_emotions(raw_text)
                probability = get_prediction_proba(raw_text)
                add_prediction_details(raw_text, prediction, np.max(probability), datetime.now())

                with col1:
                    st.success("Original Text")
                    st.write(raw_text)

                    st.success("Prediction")
                    emoji_icon = emotions_emoji_dict[prediction]
                    st.write("{}:{}".format(prediction, emoji_icon))
                    st.write("Confidence:{}".format(np.max(probability)))

                with col2:
                    st.success("Prediction Probability")
                    proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
                    proba_df_clean = proba_df.T.reset_index()
                    proba_df_clean.columns = ["emotions", "probability"]

                    fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
                    st.altair_chart(fig, use_container_width=True)

            elif selected_tab == "Sentiment Analysis":
                sentiment_prediction = predict_sentiment(raw_text)

                with col1:
                    st.success("Original Text")
                    st.write(raw_text)

                    st.success("Sentiment Prediction")
                    st.write("Sentiment: {}".format(sentiment_prediction))
                   
    elif choice == "📈Monitor":
        add_page_visited_details("Monitor", datetime.now())
        st.subheader("Monitor App")

        with st.expander("Page Metrics"):
            page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=['Page Name', 'Time of Visit'])
            st.dataframe(page_visited_details)

            pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
            c = alt.Chart(pg_count).mark_bar().encode(x='Page Name', y='Counts', color='Page Name')
            st.altair_chart(c, use_container_width=True)

            p = px.pie(pg_count, values='Counts', names='Page Name')
            st.plotly_chart(p, use_container_width=True)

        with st.expander('Emotion Classifier Metrics'):
            df_emotions = pd.DataFrame(view_all_prediction_details(), columns=['Rawtext', 'Prediction', 'Probability', 'Time_of_Visit'])
            st.dataframe(df_emotions)

            prediction_count = df_emotions['Prediction'].value_counts().rename_axis('Prediction').reset_index(name='Counts')
            pc = alt.Chart(prediction_count).mark_bar().encode(x='Prediction', y='Counts', color='Prediction')
            st.altair_chart(pc, use_container_width=True)
    elif choice == "sarcasm": 
     
        add_page_visited_details("scarsm", datetime.now())
        st.subheader("Sarcasm detection app")
        
        # Add tabs for different text analysis tasks
        
        with st.form(key='text_analysis_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            # Apply the selected text analysis function
            sar_prediction = predict_sarcasm(raw_text)
            

            
            st.success("Original Text")
            st.write(raw_text)

            st.success("Sarcasm Prediction")
            st.write("Sarcasm: {}".format(sar_prediction))                
    else:
        add_page_visited_details("About", datetime.now())

        st.write("Welcome to the Emotion and Sentiment Analysis App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions and sentiments in textual data.")

        st.subheader("Our Mission")

        st.write("At Emotion and Sentiment Analysis, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional and sentiment content hidden within text. We believe that emotions and sentiments play a crucial role in communication, and by uncovering these aspects, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")

        st.subheader("How It Works")

        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained models, which predict the emotions and sentiments associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional and sentiment content of your text.")

        st.subheader("Key Features:")

        st.markdown("##### 1. Real-time Emotion and Sentiment Analysis")

        st.write("Our app offers real-time emotion and sentiment analysis, allowing you to instantly analyze the emotions and sentiments expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions and sentiments underlying the text.")

        st.markdown("##### 2. Confidence Score")

        st.write("Alongside the detected")
		
if __name__ == '__main__':
	main()