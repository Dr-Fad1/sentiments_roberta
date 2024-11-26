import streamlit as st
import pandas as pd
from transformers import pipeline
from io import BytesIO
import time
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

    # Bar Chart
    st.write("### Sentiment Summary")
    st.bar_chart(sentiment_counts)

    # Word Cloud
    st.write("### Word Cloud of Texts")
    text_data = " ".join(df['Text'].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text_data)
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
