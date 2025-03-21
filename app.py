# Importing modules
import nltk
import streamlit as st
import re
import preprocessor  # Custom preprocessing module
import helper  # Custom helper module (if needed)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from PIL import Image
import io
import json

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
model = load_model('/Users/diksha/Downloads/lstm_model.h5')
model2=load_model('/Users/diksha/Downloads/image_model.h5')
sentiment_classes = ["Angry", "Disgusted", "fearful", "Happy", "Neutral", "Sad", "Surprised"]

import os
import shutil

import streamlit as st
import nltk
import io
from collections import Counter
import tempfile
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from pydub import AudioSegment
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from nltk.tokenize import word_tokenize
import string
import time
import tempfile
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt_tab', force=True)  # to tokenize
nltk.download('stopwords')  # to remove stopwords
nltk.download('vader_lexicon')
nltk.download('wordnet')

# Function to perform sentiment analysis
def sentiment_analyse(sentiment_text):
    score = SentimentIntensityAnalyzer().polarity_scores(sentiment_text)
    neg = score['neg']
    pos = score['pos']
    if neg > pos:
        return "Negative sentiment"
    elif pos > neg:
        return "Positive sentiment"
    else:
        return "Neutral sentiment"

# Load emotions from the file into a dictionary
emotions = {}
with open('emotions.txt', 'r') as file:
    for line in file:
        clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
        word, emotion = clear_line.split(':')
        emotions[word] = emotion.capitalize()  # Capitalize the first letter

# load model
emotion_dict = {0: 'angry', 1:'disgust', 2:'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# load json and create model
json_file = open('face_emotion.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)
# load weights into a new model
classifier.load_weights("face_emotion.weights.h5")

# load face
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


def analyze_uploaded_video(video_file):
    # Create a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        shutil.copyfileobj(video_file, temp_video_file)
        temp_video_path = temp_video_file.name
    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        st.error("Error opening video file. Please try again.")
        return

    st.write("Processing video. This may take a while...")
    sentiment_counts = {emotion: 0 for emotion in emotion_dict.values()}
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Convert to grayscale for emotion detection
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(image=img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                sentiment_counts[finalout] += 1
                # Draw label
                label_position = (x, y)
                cv2.rectangle(frame, (x, y - 25), (x + w, y), (0, 0, 0), -1)
                cv2.putText(frame, finalout.capitalize(), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the frame with detected faces and emotions
        # st.write(frame_count)
        # st.image(frame, channels="BGR")

    cap.release()

    st.success(f"Video processing completed. {frame_count} frames analyzed.")
    # st.write("### Sentiment Analysis Results")
    # for emotion, count in sentiment_counts.items():
    #     st.write(f"{emotion}: {count} frames")

    fig, ax = plt.subplots()
    ax.bar(sentiment_counts.keys(), sentiment_counts.values(), color='skyblue')
    ax.set_xlabel('Emotions')
    ax.set_ylabel('Frame Count')
    ax.set_title('Sentiment Analysis Result')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Determine and display overall sentiment
    overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    st.markdown(f"<h2 style='color:#4CAF50; font-family:sans-serif;'>Overall Sentiment: {overall_sentiment}</h2>",
                unsafe_allow_html=True)

    # Display sentiment counts with better aesthetics
    st.markdown("<h3 style='color:#2196F3;'>Detailed Sentiment Counts:</h3>", unsafe_allow_html=True)
    for emotion, count in sentiment_counts.items():
        st.markdown(f"<p style='color:#555; font-size:16px;'>{emotion}: {count} frames</p>", unsafe_allow_html=True)

    # overall_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    # st.write(f"Overall Sentiment: {overall_sentiment}")


def analyze_image_sentiment(image):
    image = image.convert('RGB')  # Ensure the image is in RGB mode (optional if needed)
    image = image.resize((48, 48))  # Resize to 48x48 as expected by the model
    image = image.convert('L')  # Convert to grayscale
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=-1)  # Add channel dimension (1 for grayscale)
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension (1, 48, 48, 1)

        # Now, the image_array is in the shape (1, 48, 48, 1), which matches the input shape expected by the model
    prediction = model2.predict(image_array)[0]
    print(prediction)
    if prediction.shape[0] == 1:
        # Use np.argmax() to find the index of the highest prediction value
        predicted_class_index = np.argmax(prediction)
        print(predicted_class_index)
        # Get the predicted sentiment from the classes
        predicted_sentiment = sentiment_classes[predicted_class_index]

        return predicted_sentiment
    else:
        return "Invalid prediction shape"

# App title
st.sidebar.title("Whatsapp Chat Sentiment Analyzer")
st.markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Chat Sentiment Analyzer</h1>",
            unsafe_allow_html=True)
# File upload button

st.sidebar.title("Select Input Type")

# Dropdown to choose input type
input_option = st.sidebar.selectbox(
    "Choose an option:",
    ["Upload Text File", "Upload Image File", "Upload Video File", "Enter Text Manually"]
)

uploaded_file = None
user_text = None
isText=False
predict_button=None
# Dropdowns for file uploads based on selection
if input_option == "Upload Text File":
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
elif input_option == "Upload Image File":
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
elif input_option == "Upload Video File":
    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
elif input_option == "Enter Text Manually":
    isText=True
    user_text = st.text_area("Enter your text here:", height=70)
    predict_button = st.button("Analyze Text")


# uploaded_files = st.file_uploader("Upload File", type=["txt", "jpg", "png", "jpeg","mp4", "avi", "mov"],
#                                   accept_multiple_files=True)
# text_sentiment = None
# image_sentiment = None
# Main heading

if uploaded_file is not None and not isText:
    # Getting byte form & then decoding
        if uploaded_file.name.endswith(".txt"):
            bytes_data = uploaded_file.getvalue()
            chat_data = bytes_data.decode("utf-8")

            # Perform preprocessing
            data = preprocessor.preprocess(chat_data)  # Apply custom preprocessing
            data['message'] = data['message'].astype(str)  # Ensure message is in string format

            # Load the pre-trained LSTM model

            # Tokenizer for the LSTM model
            tokenizer = Tokenizer(num_words=3000, oov_token='<OOV>')
            tokenizer.fit_on_texts(data['message'])  # Fit tokenizer on the messages (vocabulary)


            # Define the sentiment prediction function
            def get_sentiment(row):
                message = row['message']

                # Convert message to sequence of integers
                processed_message = tokenizer.texts_to_sequences(message)

                # Pad the sequence to the required length (200 in this case)
                padded_message = pad_sequences(processed_message, maxlen=200, padding='post')

                # Predict sentiment (output is a probability, e.g., 0.75 = positive)
                prediction = model.predict(padded_message)

                # Classify sentiment based on the predicted probability
                if prediction > 0.6:
                    return 1  # Positive sentiment
                elif prediction < 0.3:
                    return -1  # Negative sentiment
                else:
                    return 0  # Neutral sentiment


            # Apply the sentiment prediction to each row in the data
            # data['sentiment'] = data.apply(lambda row: get_sentiment(row), axis=1)

            # convert to a sequence
            sequences = tokenizer.texts_to_sequences(data['message'])
            # pad the sequence
            padded = pad_sequences(sequences, padding='post', maxlen=100)
            # Get labels based on probability 1 if p>= 0.5 else 0
            prediction = model.predict(padded)
            # print(prediction);
            pred_labels = []
            for i in prediction:
                if i > 0.6:
                    pred_labels.append(1)  # Positive sentiment
                elif i < 0.4:
                    pred_labels.append(-1)  # Negative sentiment
                else:
                    pred_labels.append(0)  # Neutral sentiment
            # sentiment_labels = [-1, 0, 1]  # Negative, Neutral, Positive
            # pred_labels = [sentiment_labels[np.argmax(i)] for i in prediction]

            # Iterate through predictions
            # pred_labels = []
            # for i in prediction:
            #     sentiment_index = np.argmax(i)  # Get index of the highest probability
            #     print(i, sentiment_index);
            #     pred_labels.append(sentiment_labels[sentiment_index])  # Assign corresponding sentiment
            #
            # print(pred_labels)

            data['value'] = pred_labels
            # User names list
            user_list = data['user'].unique().tolist()

            # Sorting
            user_list.sort()

            # Insert "Overall" at index 0
            user_list.insert(0, "Overall")

            # Selectbox
            selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

            if st.sidebar.button("Show Analysis"):
                # Monthly activity map
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",
                                unsafe_allow_html=True)

                    busy_month = helper.month_activity_map(selected_user, data, 1)

                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",
                                unsafe_allow_html=True)

                    busy_month = helper.month_activity_map(selected_user, data, 0)

                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",
                                unsafe_allow_html=True)

                    busy_month = helper.month_activity_map(selected_user, data, -1)

                    fig, ax = plt.subplots()
                    ax.bar(busy_month.index, busy_month.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                # Daily activity map
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",
                                unsafe_allow_html=True)

                    busy_day = helper.week_activity_map(selected_user, data, 1)

                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",
                                unsafe_allow_html=True)

                    busy_day = helper.week_activity_map(selected_user, data, 0)

                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",
                                unsafe_allow_html=True)

                    busy_day = helper.week_activity_map(selected_user, data, -1)

                    fig, ax = plt.subplots()
                    ax.bar(busy_day.index, busy_day.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                # Weekly activity map
                # col1, col2, col3 = st.columns(3)
                # with col1:
                #     try:
                #         st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",
                #                     unsafe_allow_html=True)
                #
                #         user_heatmap = helper.activity_heatmap(selected_user, data, 1)
                #
                #         fig, ax = plt.subplots()
                #         ax = sns.heatmap(user_heatmap)
                #         st.pyplot(fig)
                #     except:
                #         st.image('error.webp')
                # with col2:
                #     try:
                #         st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",
                #                     unsafe_allow_html=True)
                #
                #         user_heatmap = helper.activity_heatmap(selected_user, data, 0)
                #
                #         fig, ax = plt.subplots()
                #         ax = sns.heatmap(user_heatmap)
                #         st.pyplot(fig)
                #     except:
                #         st.image('error.webp')
                # with col3:
                #     try:
                #         st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",
                #                     unsafe_allow_html=True)
                #
                #         user_heatmap = helper.activity_heatmap(selected_user, data, -1)
                #
                #         fig, ax = plt.subplots()
                #         ax = sns.heatmap(user_heatmap)
                #         st.pyplot(fig)
                #     except:
                #         st.image('error.webp')

                # Daily timeline
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",
                                unsafe_allow_html=True)

                    daily_timeline = helper.daily_timeline(selected_user, data, 1)

                    fig, ax = plt.subplots()
                    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",
                                unsafe_allow_html=True)

                    daily_timeline = helper.daily_timeline(selected_user, data, 0)

                    fig, ax = plt.subplots()
                    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",
                                unsafe_allow_html=True)

                    daily_timeline = helper.daily_timeline(selected_user, data, -1)

                    fig, ax = plt.subplots()
                    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                # Monthly timeline
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",
                                unsafe_allow_html=True)

                    timeline = helper.monthly_timeline(selected_user, data, 1)

                    fig, ax = plt.subplots()
                    ax.plot(timeline['time'], timeline['message'], color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",
                                unsafe_allow_html=True)

                    timeline = helper.monthly_timeline(selected_user, data, 0)

                    fig, ax = plt.subplots()
                    ax.plot(timeline['time'], timeline['message'], color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",
                                unsafe_allow_html=True)

                    timeline = helper.monthly_timeline(selected_user, data, -1)

                    fig, ax = plt.subplots()
                    ax.plot(timeline['time'], timeline['message'], color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

                # Percentage contributed
                if selected_user == 'Overall':
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",
                                    unsafe_allow_html=True)
                        x = helper.percentage(data, 1)

                        # Displaying
                        st.dataframe(x)
                    with col2:
                        st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",
                                    unsafe_allow_html=True)
                        y = helper.percentage(data, 0)

                        # Displaying
                        st.dataframe(y)
                    with col3:
                        st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",
                                    unsafe_allow_html=True)
                        z = helper.percentage(data, -1)

                        # Displaying
                        st.dataframe(z)

                # Most Positive,Negative,Neutral User...
                if selected_user == 'Overall':
                    # Getting names per sentiment
                    x = data['user'][data['value'] == 1].value_counts().head(10)
                    y = data['user'][data['value'] == -1].value_counts().head(10)
                    z = data['user'][data['value'] == 0].value_counts().head(10)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",
                                    unsafe_allow_html=True)

                        # Displaying
                        fig, ax = plt.subplots()
                        ax.bar(x.index, x.values, color='green')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col2:
                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",
                                    unsafe_allow_html=True)

                        # Displaying
                        fig, ax = plt.subplots()
                        ax.bar(z.index, z.values, color='grey')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    with col3:
                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",
                                    unsafe_allow_html=True)

                        # Displaying
                        fig, ax = plt.subplots()
                        ax.bar(y.index, y.values, color='red')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)

                # WORDCLOUD......
                col1, col2, col3 = st.columns(3)
                with col1:
                    try:
                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",
                                    unsafe_allow_html=True)

                        # Creating wordcloud of positive words
                        df_wc = helper.create_wordcloud(selected_user, data, 1)
                        fig, ax = plt.subplots()
                        ax.imshow(df_wc)
                        st.pyplot(fig)
                    except:
                        # Display error message
                        st.image('error.webp')
                with col2:
                    try:
                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",
                                    unsafe_allow_html=True)

                        # Creating wordcloud of neutral words
                        df_wc = helper.create_wordcloud(selected_user, data, 0)
                        fig, ax = plt.subplots()
                        ax.imshow(df_wc)
                        st.pyplot(fig)
                    except:
                        # Display error message
                        st.image('error.webp')
                with col3:
                    try:
                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",
                                    unsafe_allow_html=True)

                        # Creating wordcloud of negative words
                        df_wc = helper.create_wordcloud(selected_user, data, -1)
                        fig, ax = plt.subplots()
                        ax.imshow(df_wc)
                        st.pyplot(fig)
                    except:
                        # Display error message
                        st.image('error.webp')

                # Most common positive words
                col1, col2, col3 = st.columns(3)
                with col1:
                    try:
                        # Data frame of most common positive words.
                        most_common_df = helper.most_common_words(selected_user, data, 1)

                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>",
                                    unsafe_allow_html=True)
                        fig, ax = plt.subplots()
                        ax.barh(most_common_df[0], most_common_df[1], color='green')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    except:
                        # Disply error image
                        st.image('error.webp')
                with col2:
                    try:
                        # Data frame of most common neutral words.
                        most_common_df = helper.most_common_words(selected_user, data, 0)

                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>",
                                    unsafe_allow_html=True)
                        fig, ax = plt.subplots()
                        ax.barh(most_common_df[0], most_common_df[1], color='grey')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    except:
                        # Disply error image
                        st.image('error.webp')
                with col3:
                    try:
                        # Data frame of most common negative words.
                        most_common_df = helper.most_common_words(selected_user, data, -1)

                        # heading
                        st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>",
                                    unsafe_allow_html=True)
                        fig, ax = plt.subplots()
                        ax.barh(most_common_df[0], most_common_df[1], color='red')
                        plt.xticks(rotation='vertical')
                        st.pyplot(fig)
                    except:
                        # Disply error image
                        st.image('error.webp')

        elif uploaded_file.name.lower().endswith((".jpg", ".png", ".jpeg")):  # Process image file
            pil_image = Image.open(uploaded_file)
            original_image = np.array(pil_image)
            st.image(original_image, use_container_width=True)
            if st.button("Analyze Image"):

                if original_image.ndim == 2:
                    image = original_image
                else:
                    # Convert the image to grayscale
                    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(image=image, scaleFactor=1.3, minNeighbors=5)

                if len(faces) == 0:
                    st.error("No human face detected. Please upload an image containing a human face.")
                # Resize the image to match the input size of the model
                else:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(img=original_image, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
                        face_image = image[y:y + h, x:x + w]

                        # Resize the image to match the input size of the model
                        face_image = cv2.resize(face_image, (48, 48), interpolation=cv2.INTER_AREA)

                        # Normalize the image
                        face_image = face_image.astype('float') / 255.0
                        face_image = img_to_array(face_image)
                        face_image = np.expand_dims(face_image, axis=0)

                        # Use the pre-trained model to predict emotions
                        prediction = classifier.predict(face_image)[0]
                        max_index = int(np.argmax(prediction))
                        predicted_emotion = emotion_dict[max_index]

                        # Draw a filled rectangle with a black background
                        cv2.rectangle(original_image, (x, y - 25), (x + w, y), (0, 0, 255), -1)

                        # Put white text on the black rectangle
                        cv2.putText(original_image, predicted_emotion, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                    # Display the image with detected faces and predicted emotions
                    st.image(original_image, use_container_width=True)
        elif uploaded_file.name.lower().endswith((".mp4", ".avi", ".mov")):
            # uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
            st.video(uploaded_file)
            if st.button("Analyze Video"):
                analyze_uploaded_video(uploaded_file)

else:
    if input_option == "Enter Text Manually" and user_text:

        if predict_button:
            st.write("Input Text: ", user_text)
            cleaned_text = user_text.lower().translate(str.maketrans('', '', string.punctuation))

            # Tokenize and remove stopwords
            tokenized_words = word_tokenize(cleaned_text, "english")
            final_words = [word for word in tokenized_words if word not in stopwords.words('english')]

            # Get emotions from the user input
            detected_emotions_set = set()  # Use a set to store unique emotions
            for word in final_words:
                if word in emotions:
                    detected_emotions_set.add(emotions[word])

            # Sort the detected emotions
            detected_emotions = sorted(detected_emotions_set)

            # Get sentiment analysis result
            sentiment_result = sentiment_analyse(cleaned_text)

            # Display the results using st.success
            st.success(f"Detected Sentiment: {sentiment_result}")

            # if detected_emotions:
            #     st.success("Detected Emotions: " + ", ".join(emotion.title() for emotion in detected_emotions))
            # else:
            #     st.info("No emotions detected.")