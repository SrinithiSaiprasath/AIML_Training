import streamlit as st

from main import Image
from main import Sentiment_Analyser
from main import Review_Classifier
from main import Generator
from main import Summarizer

st.title("NLP Operations")
st.write("Explore various NLP functionalities including Sentiment Analysis, Review Classification, Text Generation, and Summarization. Choose an operation, provide input, and get instant results.")

st.sidebar.title("Choose an Operation")
st.sidebar.write("Select an operation to process your text using state-of-the-art NLP techniques.")

options = {
    "Sentiment Analyser": "Analyze the sentiment of the given text to determine  its emotion",
    "Review Classifier": "Classify reviews into predefined categories based on the content into positive,negative or neutral.",
    "Text Generator": "Generate coherent and contextually relevant text based on the input provided.",
    "Summarizer": "Summarize the input text to provide a concise and meaningful summary.",
    "Caption Generator": "Generate a caption for the uploaded image or image URL."
}

# Displaying Options with Descriptions

for option, description in options.items():
    st.sidebar.subheader(option)
    st.sidebar.write(description)
    # st.sidebar.radio("" , option)


# User Selection
# Display Selected Operation and Input Area
section = st.sidebar.radio("", list(options.keys()), index=None)

if section is None:
    st.header("Choose your Operation from the Sidebar")
else:
    st.header(section)
    st.write(options[section])

# Input Section Logic
try:
    if section == "Caption Generator":
        input_mode = st.radio("Choose Image Input Mode:", ["Upload Image", "Enter Image URL"])

        if input_mode == "Upload Image":
            uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
            if uploaded_file and st.button("Submit"):
                with st.spinner("Generating caption..."):
                    result = Caption_Generator(uploaded_file)
                    st.success(result)

        elif input_mode == "Enter Image URL":
            image_url = st.text_input("Enter Image URL:")
            if image_url and st.button("Submit"):
                with st.spinner("Generating caption..."):
                    result = Caption_Generator(image_url)
                    st.success(result)

    else:
        user_input = st.text_area("Enter your text here:")
        if st.button("Submit"):
            with st.spinner("Processing..."):
                if section == "Sentiment Analyser":
                    result = Sentiment_Analyser(user_input)
                elif section == "Review Classifier":
                    result = Review_Classifier(user_input)
                elif section == "Text Generator":
                    result = Generator(user_input)
                elif section == "Summarizer":
                    result = Summarizer(user_input)
                st.success(result)

except Exception as e:
    st.error(f"An error occurred: {e}")