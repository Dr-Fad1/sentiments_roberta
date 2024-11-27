import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import random

# Ensure DIN fonts are used
from matplotlib import rcParams
rcParams['font.family'] = 'DIN'

# Load the sentiment analysis pipeline
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", device= -1)

sentiment_analyzer = load_model()

# Define sentiment mapping
sentiment_mapping = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Streamlit app
st.title("Sentiment Analysis App")

# Upload Excel file
uploaded_file = st.file_uploader("Upload an Excel file with a column called 'Text'", type=["xlsx"])

if uploaded_file:
    # Load the Excel file
    try:
        df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    # Check if the 'Text' column exists
    if "Text" not in df.columns:
        st.error("The uploaded file must contain a column named 'Text'.")
        st.stop()

    st.write("File uploaded successfully!")
    # Show the first few rows of the uploaded file for preview
    st.write("Data Preview:", df.head())
    st.write("_______________________________________________")
    st.write("Performing sentiment analysis...")

    # Start timer
    start_time = time.time()

    # Perform sentiment analysis
    df['Sentiment'] = df['Text'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
    df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

    # End timer
    end_time = time.time()
    time_taken = end_time - start_time

    st.write("Sentiment analysis complete!")
    st.write(f"Time taken: {time_taken:.2f} seconds")
    st.write("_______________________________________________")
    st.write("Data Output Preview:", df.head())

    # Sentiment Counts
    sentiment_counts = df['Sentiment'].value_counts()

    # Bar Chart with DIN font
    st.write("### Sentiment Summary")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(sentiment_counts.index, sentiment_counts.values, color="#5C4621")
    ax.set_title("Sentiment Count", fontsize=16, color="#5C4621", fontname="DIN")
    ax.set_ylabel("Count", fontsize=12, color="#5C4621", fontname="DIN")
    ax.set_xlabel("Sentiment", fontsize=12, color="#5C4621", fontname="DIN")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

    # Word Cloud with DIN font
    st.write("### Word Cloud of Texts")
    custom_stopwords = set(STOPWORDS)
    custom_stopwords.update(["Hello", "Hi", "hey", "greetings", "equity"])  # Add words to exclude
    text_data = " ".join(df['Text'].astype(str).tolist())

    # Define custom color function
    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        colors = ["#5C4621", "#A32A29", "#221F1F", "#C04F15", "#F4AB7A", "#80350E"]
        return random.choice(colors)

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        stopwords=custom_stopwords,
        color_func=custom_color_func,
        font_path='DIN-Regular.ttf'  # Update this path with the location of your DIN font
    ).generate(text_data)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # Allow the user to download the results
    def to_excel(dataframe):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            dataframe.to_excel(writer, index=False)
        processed_data = output.getvalue()
        return processed_data

    excel_data = to_excel(df)
    st.download_button(
        label="Download Results as Excel",
        data=excel_data,
        file_name="sentiment_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
