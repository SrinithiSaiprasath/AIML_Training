# Install necessary libraries
# ! pip install googletrans==4.0.0-rc1 textblob langdetect nltk transformers streamlit pytextrank
# ! python -m spacy download en_core_web_lg

# Language Detection and Translation
from langdetect import detect
from googletrans import Translator
from textblob import TextBlob

# Transformers for NLP
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM, pipeline

# TensorFlow and PyTorch
import tensorflow as tf
# import torch
# from torch import nn
# from torch.utils.data import DataLoader, Dataset
# from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

# Machine Learning and Data Science Libraries
import pandas as pd
import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report

# Visualization Libraries
import matplotlib.pyplot as plt
# import seaborn as sns

# Streamlit for Web Interface
import streamlit as st

# NLTK for Natural Language Processing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# SpaCy and PyTextRank for NLP
import torch 
import spacy

# import pytextrank

# Initialize SpaCy model
nlp = spacy.load("en_core_web_lg")

# Translator instance
translator = Translator()
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Language detection failed: {e}")
        return "unknown"

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        return text
def Review_Classifier(text):
    lang = detect_language(text)
    if lang != "en":
        text = translate_to_english(text)
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Range: [-1, 1]
    if sentiment_score > 0.1:
        return "positive"
    elif sentiment_score < -0.1:
        return "negative"
    else:
        return "neutral"

def get_GPT2tokenizer():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = TFGPT2LMHeadModel.from_pretrained("gpt2-xl")
    return model , tokenizer

def Generator(prompt):
  model , tokenizer = get_GPT2tokenizer()
  inputs_id= tokenizer.encode(prompt, return_tensors='tf' ,truncation=True)
  beam_output = model.generate(inputs_id , max_length = 250 , num_beams = 5 , no_repeat_ngram_size = 2 , early_stopping = True ,pad_token_id = tokenizer.eos_token_id , )
  output = tokenizer.decode(beam_output[0] , skip_special_tokens = True , clean_up_tokenization_spaces = True)
  return output

def detect_language(text):
    try:
        return detect(text)
    except:
        return None

def translate_to_english(text):
    try:
        translated = translator.translate(text, dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation failed: {e}")
        return text
def pre_process_text(text , lang ="english"):
  try:
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words(lang))
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_tokens)
  except:
    return text
  
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

def Sentiment_Analyser(sentence):
  sentence = translate_to_english(sentence)
  results = emotion_pipeline(sentence)
  for result in results:
    return(result['label'])

def Summarizer(input):
  doc = nlp(input)
  for sentence in doc._.textrank.summary(limit_phrases=5, limit_sentences=5):
    print(sentence.text) # Access the text