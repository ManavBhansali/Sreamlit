#pip install translate streamlit torch pyttsx3 SpeechRecognition PyAudio transformers
import os
import pyttsx3
from translate import Translator
import streamlit as st
import torch
import speech_recognition as sr
from transformers import MarianMTModel, MarianTokenizer
from queue import Queue
from threading import Thread

# Ensure temporary directory exists
os.makedirs("temp", exist_ok=True)

# Initialize the pyttsx3 engine globally
engine = pyttsx3.init()
engine.setProperty("rate", 150)  # You can adjust the rate

# Set up the recognizer
recognizer = sr.Recognizer()

# Create a queue for TTS messages
tts_queue = Queue()

# Function to handle TTS processing
def process_tts():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()

# Start the TTS processing thread
tts_thread = Thread(target=process_tts, daemon=True)
tts_thread.start()

# Load translation model
@st.cache_resource
def load_translation_model(src_lang, tgt_lang):
    model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

# Function to perform translation
def translate_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

# Function for text-to-speech
def speak(text):
    tts_queue.put(text)  # Add the text to the queue

# Streamlit UI for text-to-speech translation
st.title("Text to Speech Translator")
input_language = st.selectbox("Select input language", ["en", "es", "fr", "de"])
output_language = st.selectbox("Select output language", ["en", "es", "fr", "de"])
text = st.text_area("Enter text to translate")
display_output_text = st.checkbox("Display output text")

if st.button("Convert"):
    translator = Translator(from_lang=input_language, to_lang=output_language)
    translation = translator.translate(text)
    
    # Save the translated text to audio
    my_file_name = text[:20] if len(text) > 0 else "audio"
    audio_file_path = f"temp/{my_file_name}.mp3"
    
    engine.save_to_file(translation, audio_file_path)
    engine.runAndWait()

    # Play the audio file
    audio_file = open(audio_file_path, "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3")

    if display_output_text:
        st.markdown("## Output text:")
        st.write(translation)

# Streamlit UI for speech recognition and translation
st.title("Real-Time Speech-to-Text Converter")
st.write("Click the button below and start speaking. This application will convert your speech to text in real-time!")

# Function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        st.write("Please start speaking...")
        audio_data = recognizer.listen(source)
        st.write("Recognizing...")
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results from the service"

if st.button("Start Recording"):
    result = recognize_speech()
    st.write("**You said:** ", result)

# Multi-Speaker Speech Translation
st.title("Multi-Speaker Speech Translation")
st.write("Select languages and start speaking. The system will translate each speaker's input to the target language and output it as speech.")

# User language selection
user1_src_lang = st.sidebar.selectbox("User 1 Source Language", ["en", "es", "fr", "de"])
user1_tgt_lang = st.sidebar.selectbox("User 1 Target Language", ["es", "en", "fr", "de"])
user2_src_lang = st.sidebar.selectbox("User 2 Source Language", ["en", "es", "fr", "de"])
user2_tgt_lang = st.sidebar.selectbox("User 2 Target Language", ["es", "en", "fr", "de"])

# Load models for each user
user1_tokenizer, user1_model = load_translation_model(user1_src_lang, user1_tgt_lang)
user2_tokenizer, user2_model = load_translation_model(user2_src_lang, user2_tgt_lang)

# Capture and translate function
def capture_and_translate_speech(user_model, user_tokenizer, language_label):
    with sr.Microphone() as source:
        st.write(f"{language_label}, please start speaking...")
        audio_data = recognizer.listen(source)
        st.write("Recognizing...")
        try:
            user_text = recognizer.recognize_google(audio_data, language=language_label)
            st.write(f"**{language_label} said:** ", user_text)

            # Translate text
            translated_text = translate_text(user_model, user_tokenizer, user_text)
            st.write("**Translated text:** ", translated_text)

            # Output translated text as speech
            speak(translated_text)  # Using the global engine instance
            return translated_text
        except sr.UnknownValueError:
            st.write("Could not understand the audio")
        except sr.RequestError:
            st.write("Could not request results from the service")

# Capture and translate for each user on button click
if st.button("User 1 Speak"):
    capture_and_translate_speech(user1_model, user1_tokenizer, user1_src_lang)

if st.button("User 2 Speak"):
    capture_and_translate_speech(user2_model, user2_tokenizer, user2_src_lang)
