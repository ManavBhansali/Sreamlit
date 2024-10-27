import os
import pyttsx3
from translate import Translator
import streamlit as st
import torch

# Ensure temporary directory exists
os.makedirs("temp", exist_ok=True)

def text_to_speech(input_language, output_language, text):
    try:
        # Translate text
        translator = Translator(from_lang=input_language, to_lang=output_language)
        translation = translator.translate(text)

        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)  # You can adjust the rate

        # Define the file path
        my_file_name = text[:20] if len(text) > 0 else "audio"
        audio_file_path = f"temp/{my_file_name}.mp3"

        # Save the translated text to audio
        engine.save_to_file(translation, audio_file_path)
        engine.runAndWait()

        return my_file_name, translation
    except Exception as e:
        return None, f"Translation failed: {str(e)}"

# Streamlit UI components
st.title("Text to Speech Translator")
input_language = st.selectbox("Select input language", ["en", "es", "fr", "de"])
output_language = st.selectbox("Select output language", ["en", "es", "fr", "de"])
text = st.text_area("Enter text to translate")
display_output_text = st.checkbox("Display output text")

if st.button("Convert"):
    result, output_text = text_to_speech(input_language, output_language, text)
    if result:
        # Play the audio file
        audio_file = open(f"temp/{result}.mp3", "rb")
        audio_bytes = audio_file.read()
        st.markdown("## Your audio:")
        st.audio(audio_bytes, format="audio/mp3", start_time=0)

        if display_output_text:
            st.markdown("## Output text:")
            st.write(output_text)
    else:
        st.error("Error: " + output_text)


from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-de"  # Try a different model
try:
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    print("Model and tokenizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")


# Function to load the translation model and tokenizer
@st.cache_resource
def load_translation_model(source_lang, target_lang):
    local_path = f"./models/opus-mt-{source_lang}-{target_lang}"
    hub_path = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    try:
        tokenizer = MarianTokenizer.from_pretrained(local_path)
        model = MarianMTModel.from_pretrained(local_path)
        st.info("Loaded model from local path.")
    except Exception:
        try:
            tokenizer = MarianTokenizer.from_pretrained(hub_path)
            model = MarianMTModel.from_pretrained(hub_path)
            st.info("Loaded model from Hugging Face Hub.")
        except Exception as e:
            st.error(f"Error loading model for {source_lang} to {target_lang}: {e}")
            return None, None
    return tokenizer, model

# Function to translate text
def translate_text(text, tokenizer, model):
    if tokenizer is None or model is None:
        return "Translation model unavailable."
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return translated_text[0]

# Function to translate text to multiple target languages
def translate_multiple(text, source_lang, target_langs):
    translations = {}
    for target_lang in target_langs:
        # Skip translation if source and target languages are the same
        if source_lang == target_lang:
            translations[target_lang] = text
            continue

        # Load the translation model and tokenizer
        tokenizer, model = load_translation_model(source_lang, target_lang)
        translated_text = translate_text(text, tokenizer, model)
        translations[target_lang] = translated_text

    return translations

# Streamlit UI components
st.title("Multi-Language Text Translator")

# Input text area
input_text = st.text_area("Enter text to translate:")

# Source language selection
source_language = st.selectbox("Select Source Language", ["en", "fr", "de", "es"])

# Target language selection
target_languages = st.multiselect("Select Target Languages", ["en", "fr", "de", "es"], default=["fr"])

if st.button("Translate"):
    if input_text:
        # Perform translation
        translated_outputs = translate_multiple(input_text, source_language, target_languages)

        # Display the translations
        for lang, translation in translated_outputs.items():
            st.markdown(f"### Translation to {lang}:")
            st.write(translation)
    else:
        st.error("Please enter some text to translate.")


# Install required libraries:
# !pip install streamlit SpeechRecognition

# import streamlit as st
import speech_recognition as sr

# Set up the recognizer and microphone
recognizer = sr.Recognizer()

st.title("Real-Time Speech-to-Text Converter")
st.write("Click the button below and start speaking. This application will convert your speech to text in real-time!")

# Function to recognize speech
def recognize_speech():
    with sr.Microphone() as source:
        st.write("Please start speaking...")
        audio_data = recognizer.listen(source)
        st.write("Recognizing...")

        try:
            # Using Google Web Speech API to recognize speech
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Could not understand the audio"
        except sr.RequestError:
            return "Could not request results from the service"

# Start speech recognition on button click
if st.button("Start Recording"):
    result = recognize_speech()
    st.write("**You said:** ", result)



# Install required libraries:
# Set up recognizer for capturing audio
from transformers import MarianMTModel, MarianTokenizer
recognizer = sr.Recognizer()

# Set up text-to-speech engine
engine = pyttsx3.init()

# Load translation models
def load_translation_model(src_lang, tgt_lang):
    model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    return model, tokenizer

# Function to perform translation
def translate_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Initialize Streamlit app
st.title("Multi-Speaker Speech Translation")
st.write("Select languages and start speaking. The system will translate each speaker's input to the target language and output it as speech.")

# User 1 language selection
st.sidebar.header("User 1 Settings")
user1_src_lang = st.sidebar.selectbox("User 1 Source Language", ["en", "es", "fr", "de"])
user1_tgt_lang = st.sidebar.selectbox("User 1 Target Language", ["es", "en", "fr", "de"])

# User 2 language selection
st.sidebar.header("User 2 Settings")
user2_src_lang = st.sidebar.selectbox("User 2 Source Language", ["en", "es", "fr", "de"])
user2_tgt_lang = st.sidebar.selectbox("User 2 Target Language", ["es", "en", "fr", "de"])

# Load translation models for each user
user1_model, user1_tokenizer = load_translation_model(user1_src_lang, user1_tgt_lang)
user2_model, user2_tokenizer = load_translation_model(user2_src_lang, user2_tgt_lang)

# Function to capture speech input and translate
def capture_and_translate_speech(user_model, user_tokenizer, language_label):
    with sr.Microphone() as source:
        st.write(f"{language_label}, please start speaking...")
        audio_data = recognizer.listen(source)
        st.write("Recognizing...")

        try:
            # Recognize speech
            user_text = recognizer.recognize_google(audio_data, language=language_label)
            st.write(f"**{language_label} said:** ", user_text)

            # Translate text
            translated_text = translate_text(user_model, user_tokenizer, user_text)
            st.write("**Translated text:** ", translated_text)

            # Output translated text as speech
            engine.say(translated_text)
            engine.runAndWait()
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

