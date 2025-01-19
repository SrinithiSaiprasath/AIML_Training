import streamlit as st
from Op import Generator
from Op import Review_Classifier
from Op import Sentiment_Analyser
from Op import Summarizer

# Page Title and Description
st.title("NLP Operations")
st.write("Explore various NLP functionalities including Sentiment Analysis, Review Classification, Text Generation, and Summarization. Choose an operation, provide input, and get instant results.")

# Sidebar Navigation with Descriptions
st.sidebar.title("Choose an Operation")
st.sidebar.write("Select an operation to process your text using state-of-the-art NLP techniques.")

options = {
    "Sentiment Analyser": "Analyze the sentiment of the given text to determine  its emotion",
    "Review Classifier": "Classify reviews into predefined categories based on the content into positive,negative or neutral.",
    "Text Generator": "Generate coherent and contextually relevant text based on the input provided.",
    "Summarizer": "Summarize the input text to provide a concise and meaningful summary."
}
# section = st.sidebar.radio("", list(options.keys()))
# st.header(section)

# Displaying Options with Descriptions
for option, description in options.items():
    st.sidebar.subheader(option)
    st.sidebar.write(description)
    # st.sidebar.radio("" , option)


# User Selection

# Display Selected Operation and Input Area
section = st.sidebar.radio("", list(options.keys()) , index = None, )
if(section == None):
   st.header("Choose your Operation from the Sidebar")
else:
    st.header(section)
    st.write(options[section])

# Common Input Area
try:
    user_input = st.text_area("Enter your text here:")
    if st.button("Submit"):
        with st.spinner("Processing..."):
            if section == "Sentiment Analyser":
                result = Sentiment_Analyser(user_input)
                # result = func.analyze(user_input)
            elif section == "Review Classifier":
                result = Review_Classifier(user_input)
                # result = func.classify(user_input)
            elif section == "Text Generator":
                result = Generator(user_input)
                # result = func.generate(user_input)
            elif section == "Summarizer":
                result = Summarizer(user_input)
                # result = func.summarize(user_input)
            
            st.text_area("Output:", value=result, height=200 , disabled = True)

    else:
        st.write("Awaiting your input...")

except(AttributeError ,NameError):
   st.text_area("No input is given...Please give text input to process")

